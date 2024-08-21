import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim = 1)  # for multi-class classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


# Example input sizes
input_size = 768 + 10  # AlephBERT embedding size + 10 additional features
hidden_size = 128  # Example hidden layer size
output_size = 2  # Two classes: schizophrenia or not
batch_size = 32
num_epochs = 20

# Initialize the network
model = FeedForwardNet(input_size, hidden_size, output_size)
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


def train(features, labels):
    features_tensor = torch.tensor(features, dtype = torch.float32)
    labels_tensor = torch.tensor(labels, dtype = torch.long)

    # Create dataset
    dataset = TextDataset(features_tensor, labels_tensor)
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    # Example training loop
    for epoch in tqdm(range(num_epochs)):
        for features, labels in train_loader:  # train_loader gives batches of (features, labels)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(features)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


if __name__ == '__main__':
    train(None, None)
    # After training, use the model to predict on new samples
    with torch.no_grad():
        test_features = torch.tensor([...])  # New input features
        predictions = model(test_features)
        predicted_class = torch.argmax(predictions, dim = 1)

