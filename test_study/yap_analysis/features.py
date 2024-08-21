from functools import lru_cache

from pydantic import BaseModel, ConfigDict
from infra.yap_wrapper.yap_api import extract_yap_analysis
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from transformers import BertModel, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from hebrew_tokenizer import tokenize

# Load BERT model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
model = BertModel.from_pretrained('onlplab/alephbert-base')
device = torch.device('cpu')
model.to(device)


class YapFeaturesMetadata(BaseModel):
    tokenized_text: str
    segmented_text: str
    lemmas: str
    dep_tree: pd.DataFrame
    md_lattice: pd.DataFrame
    ma_lattice: pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed = True, extra = "allow")

    def parse_to_df(self) -> pd.DataFrame:
        data_combined = pd.concat([self.dep_tree, self.md_lattice], axis = 1)
        data_combined = data_combined.loc[:, ~data_combined.columns.duplicated()]
        data_combined["num"] = data_combined["num"].astype(int)
        return data_combined


class YapFeatureExtractor:
    def __init__(self):
        pass

    @classmethod
    @lru_cache
    def get_text_mrl_analysis(cls, text: str, **kwargs) -> YapFeaturesMetadata:
        extracted_data = extract_yap_analysis(text, **kwargs)
        return YapFeaturesMetadata.model_validate({
            "tokenized_text": extracted_data[0],
            "segmented_text": extracted_data[1],
            "lemmas": extracted_data[2],
            "dep_tree": extracted_data[3],
            "md_lattice": extracted_data[4],
            "ma_lattice": extracted_data[5],
        })

    @staticmethod
    def agg_sentence_id(df: pd.DataFrame) -> pd.DataFrame:
        df['sentence_id'] = 1
        sentence_id = 1
        for i in range(1, len(df)):
            if df.iloc[i]['num'] < df.iloc[i - 1]['num']:
                sentence_id += 1
            df.at[i, 'sentence_id'] = sentence_id
        return df

    def extract_embeddings(self, text: str, **kwargs) -> pd.DataFrame:
        """
        1. Encode Categorical Features: Each categorical feature (word, pos, gen, etc.) will be encoded. For this,
            one-hot encoding can be used, or the categorical data can be directly embedded.

        2. Combine Encoded Features: For each word in a sentence, combine all its encoded features into a single vector.

        3. Aggregate Word Vectors into Sentence Embedding: Aggregate these word-level vectors into a sentence-level
            embedding by methods like averaging, max pooling, or a more advanced model.

        :param text:
        :param kwargs:
        :return: pd.DataFrame
        """
        # df = self.get_text_analysis(text, **kwargs)
        # df = self.agg_sentence_id(df)
        df = pd.read_csv("to_group.csv", index_col = False)
        # Grouping by sentence_id
        grouped_sentences = df.groupby('sentence_id').apply(lambda x: x)

        # Select the columns to encode
        categorical_columns = ['word', 'pos', 'pos_2', 'gen', 'tense']

        # Apply one-hot encoding
        one_hot_encoder = OneHotEncoder(sparse_output = False)
        one_hot_encoded = one_hot_encoder.fit_transform(grouped_sentences[categorical_columns].fillna('missing'))

        # Combine the one-hot encoded features with the original data
        encoded_df = pd.DataFrame(one_hot_encoded, columns = one_hot_encoder.get_feature_names_out(categorical_columns))

        # Combine with sentence_id and num columns
        final_df = pd.concat([df[['sentence_id', 'num']], encoded_df], axis = 1)
        sentence_embeddings = final_df.groupby('sentence_id').mean()
        return sentence_embeddings

    def train(self, sentence_embeddings):
        sentence_embeddings['schizophrenia'] = 1
        # Splitting the data into train and test sets
        X = sentence_embeddings.drop('schizophrenia', axis = 1)
        y = sentence_embeddings['schizophrenia']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

        # Building a Random Forest Classifier as an example
        clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
        clf.fit(X_train, y_train)

        # Predicting on the test set
        y_pred = clf.predict(X_test)

        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        accuracy, report


if __name__ == '__main__':
    a = YapFeatureExtractor()
    data = 'אישה שלומדת שיש לה רצון ללמוד איזשהם מקצועות לעומת כשהיא בפרונט ומאחוריה יש גבר ואישה וסוס שעדיין ממשיכים לטפל בחזה לא בחזה בחווה. לטפל בחווה ולהיכנס לבנייה והיא כאילו רוצה לצאת מהמעגל הזה ולהסתדר בכוחות עצמה, כי יש לה פה ספרים וכזה שלומדים וזה.'
    text_embeddings = a.extract_embeddings(text = data)
    text_embeddings.to_csv("example_embedd.csv")
    # df = pd.read_csv("../data.csv", index_col = False)
    # for _, row in df.iterrows():
