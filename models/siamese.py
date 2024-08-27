import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from test_study.coherence.features import TextFeatureExtractor


# Define the embedding model
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.dense1 = nn.Linear(768, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.dense3 = nn.Linear(64, 128)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.batch_norm1(x)
        x = F.relu(self.dense2(x))
        x = self.dropout2(x)
        x = self.dense3(x)

        # Normalize output embeddings
        x = F.normalize(x, p = 2, dim = 1)
        return x

    def predict(self, inputs):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            return self.forward(inputs)

    @classmethod
    def save_model(cls, path: str):
        embedding_model = cls()
        torch.save(embedding_model.state_dict(), path)

    @classmethod
    def load_model(cls, path: str):
        loaded_model = cls()
        state_dict = torch.load(path)
        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()
        print("Model loaded and ready for inference or further training")
        return loaded_model


# Custom Triplet Loss Layer
class TripletLossLayer(nn.Module):
    def __init__(self, alpha):
        super(TripletLossLayer, self).__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        p_dist = torch.sum((anchor - positive) ** 2, axis = 1)
        n_dist = torch.sum((anchor - negative) ** 2, axis = 1)
        loss = F.relu(p_dist - n_dist + self.alpha)
        return torch.mean(loss)


# Define the complete model that includes embeddings and triplet loss
class TripletModel(nn.Module):
    def __init__(self, embedding_model, alpha):
        super(TripletModel, self).__init__()
        self.embedding_model = embedding_model
        self.triplet_loss_layer = TripletLossLayer(alpha)

    def forward(self, anchor_input, positive_input, negative_input):
        anchor_embedding = self.embedding_model(anchor_input)
        positive_embedding = self.embedding_model(positive_input)
        negative_embedding = self.embedding_model(negative_input)
        loss = self.triplet_loss_layer(anchor_embedding, positive_embedding, negative_embedding)
        return loss


class TripletTextDataset(Dataset):
    def __init__(self, text_data, labels, embed_func):
        self.text_data = text_data
        self.labels = labels
        self.embed_func = embed_func

        # Creating a map of indices for each label
        self.label_indices = {label: np.where(np.array(labels) == label)[0] for label in np.unique(labels)}

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        # Anchor text and its label
        anchor_text = self.text_data[idx]
        anchor_label = self.labels[idx]

        # Finding a positive example with the same label as the anchor
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = np.random.choice(self.label_indices[anchor_label])

        # Finding a negative example with a different label
        negative_label = 1 - anchor_label  # Since it's binary, the other label is simply `1 - anchor_label`
        negative_idx = np.random.choice(self.label_indices[negative_label])

        # Convert text to embeddings
        anchor_embedding = self.embed_func([anchor_text]).squeeze(0)
        positive_embedding = self.embed_func([self.text_data[positive_idx]]).squeeze(0)
        negative_embedding = self.embed_func([self.text_data[negative_idx]]).squeeze(0)

        return anchor_embedding, positive_embedding, negative_embedding


def embed_func(text_batch):
    # Convert text to embeddings (e.g., using BERT)
    embeddings = [TextFeatureExtractor.get_sentence_embedding(text) for text in text_batch]
    # Convert to a tensor of shape (batch_size, embedding_dim)
    return torch.stack(embeddings)


def train():
    # Initialize the embedding and triplet models
    embedding_model = EmbeddingModel()
    alpha = 0.4
    triplet_model = TripletModel(embedding_model, alpha)

    train_text_data = train['answer'].values.tolist()
    train_labels = train['label'].values.tolist()

    # Initialize the dataset and DataLoader
    train_dataset = TripletTextDataset(train_text_data, train_labels, embed_func)
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)

    # Initialize optimizer
    optimizer = optim.Adam(triplet_model.parameters(), lr = 0.001)

    # Training Loop
    num_epochs = 100

    for epoch in tqdm(range(num_epochs)):
        triplet_model.train()  # Set model to training mode

        running_loss = 0.0
        for i, (anchor, positive, negative) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients

            loss = triplet_model(anchor, positive, negative)  # Forward pass to compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimization step

            running_loss += loss.item()

            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print('Finished Training')

