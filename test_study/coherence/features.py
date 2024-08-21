from collections import defaultdict

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
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


class TextFeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.yap_feature = YapFeatureExtractor()
        self.count_vec = CountVectorizer()
        self.tfidf_vec = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.lda = LatentDirichletAllocation(random_state = 42)
        self.model_name = model_name

    @staticmethod
    def clean_text(text: str):
        cleaned_text = re.sub(r'\[.*?\]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    @classmethod
    def get_sentence_embedding(cls, sentence):
        inputs = tokenizer(sentence, return_tensors = "pt", padding = True, truncation = True)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        return cls_embedding.squeeze().numpy()

    def calculate_coherence_score(self, sentences):
        embeddings = [self.get_sentence_embedding(sentence) for sentence in sentences]
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]
        return np.mean(similarities) if similarities else 1.0

    def calculate_attention_scores(self, sentences):
        embeddings = [self.get_sentence_embedding(sentence.strip()) for sentence in sentences]
        num_sentences = len(embeddings)

        logic_scores = []

        for i in range(1, num_sentences):
            attention_weights = []
            for j in range(i):
                attention_weight = cosine_similarity(embeddings[i], embeddings[j])[0][0]
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
                if column not in {"label", "file_name"}:
                    record = {
                        "answer": self.clean_text(value),
                        "question": column,
                        "label": row["label"],
                        "person": row["file_name"]
                    }
                    all_data.append(record)
        return pd.DataFrame(all_data)

    def encode_labels(self, df: pd.DataFrame, categories: list[str] = CATEGORIES):
        for category in categories:
            df[category] = self.label_encoder.fit_transform(df[category])
        return df

    def extract_all_features(self, text: str):
        yap_analysis = self.yap_feature.get_text_mrl_analysis(text)
        md_dep_tree_df = yap_analysis.parse_to_df()
        md_dep_tree_df = self.agg_sentence_id(md_dep_tree_df)
        grouped_sentences = md_dep_tree_df.groupby('sentence_id').apply(lambda x: x)
        sentences_id_map = defaultdict(list)
        for _, row in grouped_sentences.iterrows():
            sentences_id_map[row["sentence_id"]].append(row["lemma"])
        sentences = [' '.join(words) for id, words in sentences_id_map.items()]
        yap_analysis.attention_scores_per_sentence = self.calculate_attention_scores(sentences)
        # combin each word with its pos
        md_dep_tree_df["combined_word_pos"] = md_dep_tree_df.apply(lambda x: (x["lemma"], x["pos"]), axis = 1)
        yap_analysis.tf_idf_seg_txt = self.extract_tf_idf(yap_analysis.segmented_text)
        yap_analysis.tf_idf_tok_txt = self.extract_tf_idf(yap_analysis.tokenized_text)
        yap_analysis.lda_seg_txt = self.extract_lda(yap_analysis.tf_idf_seg_txt)
        yap_analysis.lda_tok_txt = self.extract_lda(yap_analysis.tf_idf_tok_txt)
        yap_analysis.count_vec = self.extract_count_vectorizer(yap_analysis.tokenized_text)
        return yap_analysis

    def preprocess_data(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            if row["label"] == 1:
                all_text_features = self.extract_all_features(row["answer"])
        return df


if __name__ == '__main__':
    # Apply the function to add coherence scores
    feat_extractor = TextFeatureExtractor()
    df = pd.read_csv("/Users/seanlavi/dev/Schizophrenic_Speech/data.csv", index_col = False)
    df = feat_extractor.transform_data_to_train_schema(df)
    df = feat_extractor.preprocess_data(df)
    df.to_csv("data_coherence.csv", index = False)
    pass
