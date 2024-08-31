import os
import sys
import pandas as pd
from torch.nn import CosineSimilarity
import numpy as np

from feature_extraction.morphological import MorphologicalFeatureExtractor
from infra.data_loader import DataLoader
# Get the parent directory of the current file (notebook)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
CLEAN_DATA_PATH = "data/clean_data.csv"
GROUND_TRUTH_DOCX_PATH = "/Users/seanlavi/dev/Schizophrenic_Speech/data/תיאור התמונות.docx"


class GroundTruthDistance:
    def __init__(self):
        self.data = pd.read_csv("/Users/seanlavi/dev/Schizophrenic_Speech/data/clean_data.csv", index_col = False)
        self.ground_truth_docx_path = GROUND_TRUTH_DOCX_PATH
        self.data_loader = DataLoader('')
        self.feat_extractor = MorphologicalFeatureExtractor()
        self.cosine_similarity = CosineSimilarity(dim = 0, eps = 1e-6)

    def create_ground_truth_df(self):
        content = self.data_loader._read_docx(self.ground_truth_docx_path)
        data = self.data_loader._extract_data(content)
        ground_truth_df = pd.DataFrame([data])
        return ground_truth_df

    def create_embeddings_df(self, original_df):
        embeddings_df = pd.DataFrame()

        for col in original_df.columns:
            embeddings_df[col] = original_df[col].apply(self.feat_extractor.get_sentence_embedding)

        return embeddings_df

    def calculate_ground_truth_distances(self):
        ground_truth_df = self.create_ground_truth_df()
        ground_truth_embeddings = self.create_embeddings_df(ground_truth_df)

        relevat_columns = [col for col in self.data.columns if 'תמונה' in col]
        data_embeddings = self.create_embeddings_df(self.data[relevat_columns])

        distances_df = pd.DataFrame(columns = list(data_embeddings.columns) + ["file_name"])
        distances_df['label'] = self.data['label']
        distances_df['file_name'] = self.data['file_name']

        for col in distances_df:
            if col not in {'label', 'file_name'}:
                ground_truth_embedding = np.reshape(ground_truth_embeddings[col].iloc[0], [-1, 1])
                distances_df[col] = data_embeddings[col].apply(
                    lambda x: self.cosine_similarity(ground_truth_embedding, np.reshape(x, [-1, 1])).item())
        return distances_df


if __name__ == '__main__':
    distance_calculator = GroundTruthDistance()
    distances = distance_calculator.calculate_ground_truth_distances()
    distances.to_csv('data/ground_truth_distances.csv', index = False)
