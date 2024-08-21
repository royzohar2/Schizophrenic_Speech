import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import torch

model_name = "onlplab/alephbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding for classification tasks
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze()


# Example usage
text = "הייתה ילדה שבאה מהבית ספר ואז אמא שלה באה ושאלה אותה 'איך היה?'"
embedding = get_embedding(text)


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss


# Initialize the Triplet Loss
triplet_loss = TripletLoss(margin = 1.0)

# Example embeddings (randomly generated here, replace with real embeddings)
anchor_embedding = torch.randn(128)  # 128-dim AlephBERT embedding for anchor
positive_embedding = torch.randn(128)  # 128-dim AlephBERT embedding for positive
negative_embedding = torch.randn(128)  # 128-dim AlephBERT embedding for negative

# Compute the loss
loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

for epoch in range(num_epochs):
    for batch in train_loader:  # train_loader provides triplets
        anchor, positive, negative = batch
        optimizer.zero_grad()

        # Get embeddings for each text in the triplet
        anchor_embedding = get_embedding(anchor)
        positive_embedding = get_embedding(positive)
        negative_embedding = get_embedding(negative)

        # Calculate Triplet Loss
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
