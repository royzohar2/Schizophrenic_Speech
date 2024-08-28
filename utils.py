import re

import pandas as pd


def clean_text(text):
    """Clean text by removing all occurrences of single and double square brackets and text inside them."""
    # Remove text within single or double square brackets
    return re.sub(r'\[\[.*?\]\]|\[.*?\]', '', text).strip() if isinstance(text, str) else text


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
