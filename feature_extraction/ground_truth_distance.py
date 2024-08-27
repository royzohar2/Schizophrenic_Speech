from data_loader import DataLoader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from test_study.coherence.features import TextFeatureExtractor


class GroundTruthDistance:
    def __init__(self):
        self.data = pd.read_csv("Data/clean_data.csv")
        self.ground_truth_docx_path = "Data/תיאור התמונות.docx"
        self.data_loader = DataLoader('')
        self.feat_extractor = TextFeatureExtractor()


    def create_ground_truth_df(self):
        content = self.data_loader._read_docx('Data/תיאור התמונות.docx')
        data = self.data_loader._extract_data(content)
        ground_truth_df = pd.DataFrame([data])
        return ground_truth_df
    
    def create_embeddings_df(self, original_df):
        embeddings_df = pd.DataFrame()

        for col in original_df.columns:
            embeddings_df[col] = original_df[col].apply(self.feat_extractor.get_sentence_embedding)

        return embeddings_df

