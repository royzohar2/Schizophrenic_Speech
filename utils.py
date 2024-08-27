import re

import pandas as pd
import torch

from feature_extraction.morphological import MorphologicalFeatureExtractor


def embed_func(text_batch):
    # Convert text to embeddings (e.g., using BERT)
    embeddings = [MorphologicalFeatureExtractor.get_sentence_embedding(text) for text in text_batch]
    # Convert to a tensor of shape (batch_size, embedding_dim)
    return torch.stack(embeddings)


def clean_text(text):
    """Clean text by removing all occurrences of single and double square brackets and text inside them."""
    # Remove text within single or double square brackets
    return re.sub(r'\[\[.*?\]\]|\[.*?\]', '', text).strip()


def explode_df_to_single_record(df: pd.DataFrame):
    all_data = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for column, value in row_dict.items():
            if column not in {"label", "file_name"} and value:
                record = {
                    "answer": clean_text(value),
                    "question": column,
                    "label": row["label"],
                    "person": row["file_name"]
                }
                all_data.append(record)
    transformed_df = pd.DataFrame(all_data)
    return transformed_df
