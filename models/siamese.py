import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from transformers import AutoTokenizer, AutoModel

from test_study.coherence.features import TextFeatureExtractor

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available. Using MPS for training.")
else:
    device = torch.device("cpu")
    print("MPS device is not available. Using CPU for training.")

model_name = "onlplab/alephbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
alephbert_model = AutoModel.from_pretrained(model_name)
alephbert_model.to(device)


class AlephBERTEmbeddingModel(nn.Module):
    def __init__(self, alephbert_model):
        super().__init__()
        self.alephbert_model = alephbert_model
        self.dense1 = nn.Linear(768, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.dense3 = nn.Linear(64, 128)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from AlephBERT
        outputs = self.alephbert_model(input_ids = input_ids, attention_mask = attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token embeddings

        # Apply additional dense layers and normalization
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.batch_norm1(x)
        x = F.relu(self.dense2(x))
        x = self.dropout2(x)
        x = self.dense3(x)
        x = F.normalize(x, p = 2, dim = 1)
        return x


class TripletTextDataset(Dataset):
    def __init__(self, text_data, labels, tokenizer):
        self.text_data = text_data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        # Tokenize the text
        encoded_input = self.tokenizer(self.text_data[idx], padding = 'max_length', truncation = True,
                                       return_tensors = 'pt')
        input_ids = encoded_input['input_ids'].squeeze(0)
        attention_mask = encoded_input['attention_mask'].squeeze(0)

        label = torch.tensor(self.labels[idx], dtype = torch.long)
        return input_ids, attention_mask, label


def embed_func(text_batch):
    # Convert text to embeddings (e.g., using BERT)
    embeddings = [TextFeatureExtractor.get_sentence_embedding(text) for text in text_batch]
    # Convert to a tensor of shape (batch_size, embedding_dim)
    return torch.stack(embeddings)


def train_and_evaluate(train_text_data, train_labels, test_text_data, test_labels):
    # Initialize tokenizer and datasets
    train_dataset = TripletTextDataset(train_text_data, train_labels, tokenizer)
    test_dataset = TripletTextDataset(test_text_data, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    # Initialize the AlephBERT-based embedding model
    embedding_model = AlephBERTEmbeddingModel(alephbert_model)

    # Initialize triplet margin loss and miner
    margin = 1.0
    triplet_loss = TripletMarginLoss(margin = margin)
    miner = TripletMarginMiner(margin = margin, type_of_triplets = "hard")

    optimizer = optim.Adam(embedding_model.parameters(), lr = 0.01)

    # Training Loop
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        embedding_model.train()
        running_loss = 0.0

        # Training Phase
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()

            # Get embeddings
            embeddings = embedding_model(input_ids = input_ids, attention_mask = attention_mask)

            # Mine hard triplets
            hard_triplets = miner(embeddings, labels)

            # Compute triplet loss
            loss = triplet_loss(embeddings, labels, hard_triplets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}")

        # Evaluation Phase
        embedding_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                embeddings = embedding_model(input_ids = input_ids, attention_mask = attention_mask)
                hard_triplets = miner(embeddings, labels)
                loss = triplet_loss(embeddings, labels, hard_triplets)
                test_loss += loss.item()

        print(f"Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.4f}")

    print('Finished Training and Evaluation')

    # Save the models
    torch.save(embedding_model.state_dict(), "/models/experiments/embedding_model_hard_mining.pth")
    print("Model saved successfully.")


if __name__ == '__main__':
    feat_extractor = TextFeatureExtractor()
    data_df = pd.read_csv("../data/clean_data.csv", index_col = False)
    data_df = feat_extractor.transform_data_to_train_schema(data_df).dropna()
    train_df, test_df = train_test_split(data_df, test_size = 0.2, random_state = 42)

    # Extract training data
    train_text_data = train_df['answer'].values.tolist()
    train_labels = train_df['label'].values.tolist()

    # Extract test data
    test_text_data = test_df['answer'].values.tolist()
    test_labels = test_df['label'].values.tolist()
    train_and_evaluate(train_text_data, train_labels, test_text_data, test_labels)
