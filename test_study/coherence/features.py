from collections import defaultdict

import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from test_study.yap_analysis.features import YapFeatureExtractor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re

MODEL_NAME = "onlplab/alephbert-base"  # Use the AlephBERT base model
CATEGORIES = ["pos", "dependency_part", "gen", "tense", "per", "num_s_p"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # Handles numpy scalar types like int64, float32, etc.
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(element) for element in obj]
    else:
        return obj


class TextFeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.yap_feature = YapFeatureExtractor()
        self.count_vec = CountVectorizer()
        self.tfidf_vec = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.lda = LatentDirichletAllocation(random_state = 42)
        self.model_name = model_name
        self.cosine_similarity = nn.CosineSimilarity(dim = 0, eps = 1e-6)

    @staticmethod
    def clean_text(text: str):
        if isinstance(text, str):
            cleaned_text = re.sub(r'\[.*?\]', '', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            return cleaned_text

    @classmethod
    def mean_pooling(cls, model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min = 1e-9)

    @classmethod
    def get_sentence_embedding(cls, sentence):
        encoded_input = tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = cls.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def calculate_attention_scores(self, sentences):
        embeddings = [np.reshape(self.get_sentence_embedding(sentence.strip()), [-1, 1]) for sentence in sentences]
        num_sentences = len(embeddings)

        logic_scores = []

        for i in range(1, num_sentences):
            attention_weights = []
            for j in range(i):
                attention_weight = self.cosine_similarity(embeddings[i], embeddings[j]).item()
                attention_weights.append(attention_weight)

            logic_score = np.mean(attention_weights)
            logic_scores.append(logic_score)

        return logic_scores

    def extract_count_vectorizer(self, text):
        x_count = self.count_vec.fit_transform([text])
        print(f'CountVectorizer: {x_count}')
        vec_pair = [(name, score) for name, score in zip(self.count_vec.get_feature_names_out(), x_count.toarray()[0])]
        return vec_pair

    def extract_tf_idf(self, text):
        '''
            TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that reflects the importance
            of a word in a document or corpus
        '''
        x_tfidf = self.tfidf_vec.fit_transform([text])
        print(f'TF-IDF: {x_tfidf}')
        vec_pair = [(name, score) for name, score in zip(self.tfidf_vec.get_feature_names_out(), x_tfidf.toarray()[0])]
        return vec_pair

    def extract_lda(self, x_tfidf):
        '''
            LDA is commonly used in applications like topic modeling, document clustering, and information retrieval
        '''
        x_lda = self.lda.fit_transform(np.reshape([tf_idf[1] for tf_idf in x_tfidf], [1, -1]))
        print(f'LDA:{x_lda}')
        return x_lda

    @staticmethod
    def agg_sentence_id(df: pd.DataFrame) -> pd.DataFrame:
        sentence_id = 1
        sentence_ids = [1]
        for i in range(1, len(df)):
            if df.iloc[i]['num'] < df.iloc[i - 1]['num']:
                sentence_id += 1
            sentence_ids.append(sentence_id)
        df["sentence_id"] = sentence_ids
        return df

    def transform_data_to_train_schema(self, df: pd.DataFrame):
        all_data = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            for column, value in row_dict.items():
                if column not in {"label", "file_name"} and value:
                    record = {
                        "answer": self.clean_text(value),
                        "question": column,
                        "label": row["label"],
                        "person": row["file_name"]
                    }
                    all_data.append(record)
        transformed_df = pd.DataFrame(all_data)
        transformed_df["question"] = self.label_encoder.fit_transform(transformed_df["question"])
        return transformed_df

    def encode_labels(self, df: pd.DataFrame, categories: list[str] = CATEGORIES):
        for category in categories:
            df[category] = self.label_encoder.fit_transform(df[category].astype(str))
        return df

    @staticmethod
    def aggregate_features(group):
        # Aggregate each feature using mean
        return group[CATEGORIES + ["dependency_arc"]].values

    def extract_all_features(self, text: str):
        yap_analysis = self.yap_feature.get_text_mrl_analysis(text)
        md_dep_tree_df = yap_analysis.parse_to_df()
        md_dep_tree_df = self.encode_labels(md_dep_tree_df)

        aggregated_word_feat_df = md_dep_tree_df.apply(self.aggregate_features, axis = 1).reset_index()
        sentences = self.yap_feature.yap_api_provider.split_text_to_sentences(yap_analysis.tokenized_text)
        # combin each word with its pos
        all_features = {}
        all_features["attention_scores_per_sentence"] = self.calculate_attention_scores(sentences)
        all_features["tf_idf_seg_txt"] = self.extract_tf_idf(yap_analysis.segmented_text)
        all_features["tf_idf_tok_txt"] = self.extract_tf_idf(yap_analysis.tokenized_text)
        all_features["lda_seg_txt"] = self.extract_lda(all_features["tf_idf_seg_txt"]).tolist()
        all_features["lda_tok_txt"] = self.extract_lda(all_features["tf_idf_tok_txt"]).tolist()
        all_features["agg_yap_features_vec"] = np.concatenate(aggregated_word_feat_df[0]).tolist()
        all_features["count_vec"] = self.extract_count_vectorizer(yap_analysis.tokenized_text)
        yap_analysis_dict = yap_analysis.model_dump(mode = "json", exclude = ["dep_tree", "md_lattice", "ma_lattice"])
        all_features.update(yap_analysis_dict)
        return all_features

    def preprocess_data(self, df: pd.DataFrame):
        all_data = []
        for _, row in df.iterrows():
            if row["label"] == 1:
                all_text_features = self.extract_all_features(row["answer"])
                row_dict = row.to_dict()
                row_dict.update(all_text_features)
                all_data.append(row_dict)
        return all_data


if __name__ == '__main__':
    # Apply the function to add coherence scores
    feat_extractor = TextFeatureExtractor()
    df = pd.read_csv("/Users/seanlavi/dev/Schizophrenic_Speech/data.csv", index_col = False)
    df = feat_extractor.transform_data_to_train_schema(df)
    df = feat_extractor.preprocess_data(df)
    df.to_csv("data_coherence.csv", index = False)
    pass
