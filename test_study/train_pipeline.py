import numpy as np

from models.aleph_bert import AlephBERTFineTuner, FeatureFusion
from test_study.coherence.features import TextFeatureExtractor, CATEGORIES


class TextClassificationPipeline:
    def __init__(self):
        self.feature_extractor = TextFeatureExtractor()
        self.model_trainer = AlephBERTFineTuner()

    def run_pipeline(self, df):
        df = self.feature_extractor.encode_labels(df, CATEGORIES)
        aggregated_df = self.feature_extractor.aggregate_sentence_level_features(df)
        sentence_embeddings = np.array(
            [self.feature_extractor.get_sentence_embedding(text) for text in aggregated_df['text']])
        coherence_scores = np.array(
            [self.feature_extractor.calculate_coherence_score([text]) for text in aggregated_df['text']])
        additional_features = np.column_stack((aggregated_df[CATEGORIES], coherence_scores))
        combined_features = FeatureFusion().combine_features(sentence_embeddings, additional_features)

        # Assuming you have the train/test split
        self.model_trainer.train_model(train_data, test_data)

# Example usage:
# pipeline = TextClassificationPipeline()
# pipeline.run_pipeline(df)
