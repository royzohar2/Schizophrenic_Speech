import re
from collections import defaultdict

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel
import numpy as np
from hebrew_tokenizer import tokenize

from test_study.yap_analysis.features import YapFeatureExtractor
from yap_api import extract_yap_analysis

model_name = "onlplab/alephbert-base"  # Use the AlephBERT base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)  # Binary classification
yap_feature = YapFeatureExtractor()
device = torch.device('cpu')
model.to(device)


# Function to count words in a sentence
def word_count(sentence):
    # Use regular expressions to find Hebrew words
    words = re.findall(r'\b[\u0590-\u05FF\w]+\b', sentence)
    # Return the count of words
    return len(words)


# Function to extract average word count usage for each person in the DataFrame
def extract_avg_words_use(df: pd.DataFrame):
    avg_word_counts = []

    for _, row in df.iterrows():
        total_words = 0
        count_columns = 0

        for col, data in row.items():
            if isinstance(data, str):  # Ensure the data is a string
                total_words += word_count(data)
                count_columns += 1

        avg_words = total_words / count_columns if count_columns > 0 else 0
        avg_word_counts.append(avg_words)

    # Add the average word count as a new column to the DataFrame
    df['avg_word_count'] = avg_word_counts
    return df


def tokenize_sentence(text: str):
    if not isinstance(text, str):
        text = str(text)
    # Split text into sentences using Hebrew punctuation
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]

    all_words = []

    # Tokenize each sentence individually
    for sentence in sentences:
        tokens = tokenize(sentence)
        # Extract only the actual words and relevant tokens
        words = [token for grp, token, token_num, (start_index, end_index) in tokens
                 if grp in {'HEBREW', 'PUNCTUATION', 'DATE', 'HOUR'}]
        all_words.append(words)

    return all_words


# Function to generate AlephBERT embeddings for a sentence
def sentence_embedding(sentence: str):
    model_name = "onlplab/alephbert-base"  # or any BERT-based model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(sentence, return_tensors = 'pt', padding = True, truncation = True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


# Function to calculate coherence score
def coherence_score(embeddings: list):
    if len(embeddings) < 2:
        return 1.0  # Single sentence or empty text is considered fully coherent

    similarities = []
    for i in range(1, len(embeddings)):
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])[0][0]
        similarities.append(similarity)

    # Coherence score: mean of similarities
    return np.mean(similarities) if similarities else 1.0  # High score = more coherent


def calculate_attention_scores(sentences):
    embeddings = [sentence_embedding(sentence.strip()) for sentence in sentences]
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


def add_logic_consistency_scores(df: pd.DataFrame):
    for col in df.columns:
        df[f'logic_consistency_{col}'] = df[col].apply(lambda text: calculate_attention_scores(tokenize_sentence(text)))
        df[f'logic_consistency_avg_{col}'] = df[f'logic_consistency_{col}'].apply(
            lambda scores: np.mean(scores) if scores else 1.0)
    return df


# Apply coherence score calculation across all columns
def add_coherence_scores(df: pd.DataFrame):
    for col in df.columns:
        score_col_name = f'coherence_score_{col}'
        df[score_col_name] = df[col].apply(coherence_score)
    return df


def train(train_data, test_data, train_id=1):
    training_args = TrainingArguments(
        output_dir = "./results",
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 10,
        weight_decay = 0.01,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_data,
        eval_dataset = test_data,
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained(f"./finetuned_alephbert_{train_id}")
    tokenizer.save_pretrained(f"./finetuned_alephbert_{train_id}")


def agg_sentence_id(df: pd.DataFrame) -> pd.DataFrame:
    sentence_id = 1
    sentence_ids = [1]
    for i in range(1, len(df)):
        if df.iloc[i]['num'] < df.iloc[i - 1]['num']:
            sentence_id += 1
        sentence_ids.append(sentence_id)
    df["sentence_id"] = sentence_ids
    return df


def extract_all_features(text: str):
    yap_analysis = yap_feature.get_text_mrl_analysis(text)
    md_dep_tree_df = yap_analysis.parse_to_df()
    md_dep_tree_df = agg_sentence_id(md_dep_tree_df)
    sentences_id_map = defaultdict(list)
    for _, row in md_dep_tree_df.iterrows():
        sentences_id_map[row["sentence_id"]].append(row["lemma"])
    sentences = [' '.join(words) for id, words in sentences_id_map.items()]
    yap_analysis.attention_scores_per_sentence = calculate_attention_scores(sentences)
    return yap_analysis


def label_encoding(categories):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(categories)
    df = pd.DataFrame({'Category': categories, 'Encoded_Labels': encoded_labels})


def preprocess_data(df: pd.DataFrame):
    # one_hot_encoder = OneHotEncoder(sparse_output = False)
    # one_hot_encoded = one_hot_encoder.fit_transform(df["question"].fillna('missing'))
    for _, row in df.iterrows():
        if row["label"] == 1:
            all_text_features = extract_all_features(row["answer"])
    return df


def transform_data_to_train_schema(df: pd.DataFrame):
    all_data = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for column, value in row_dict.items():
            if column not in {"label", "file_name"}:
                record = {
                    "answer": value,
                    "question": column,
                    "label": row["label"],
                    "person": row["file_name"]
                }
                all_data.append(record)
    return pd.DataFrame(all_data)


if __name__ == '__main__':
    # Apply the function to add coherence scores
    df = pd.read_csv("/Users/seanlavi/dev/Schizophrenic_Speech/data.csv", index_col = False)
    df = transform_data_to_train_schema(df)
    df = preprocess_data(df)
    df_with_logic_scores = add_logic_consistency_scores(df)
    df = add_coherence_scores(df)
    df.to_csv("data_coherence.csv", index = False)
    pass
