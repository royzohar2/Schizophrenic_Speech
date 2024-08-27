import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8  # an arbitrary small value to be used for numerical stability tricks

""" 
        Triplet loss is computed on a set of three examples:

        Anchor (A): The base example.
        Positive (P): An example that is similar to the anchor.
        Negative (N): An example that is dissimilar to the anchor. 
        
        The goal is to make the distance between A and P smaller than 
        the distance between A and N by a margin α.
        
                L(A, P, N) = max(d(A,P) − d(A,N) + α, 0)
                
        To make training more efficient and effective, "hard triplet mining" is often used, 
        which selects the most difficult triplets (where the negative is close to the anchor 
        or the positive is far from the anchor) for training. 
        This ensures that the model is always learning from the most challenging examples, 
        improving its robustness.

        Reference: https://omoindrot.github.io/triplet-loss#offline-and-online-triplet-mining
        
        1. batch all: select all the valid triplets, and average the loss on the hard and semi-hard triplets.
            a crucial point here is to not take into account the easy triplets (those with loss 0), 
            as averaging on them would make the overall loss very small this produces a total of PK (K−1) (PK−K)
            triplets (PK anchors, K − 1 possible positives per anchor, PK − K possible negatives)
            
        2. batch hard: for each anchor, select the hardest positive (biggest distance d(a,p)) 
            and the hardest negative among the batch this produces PK triplets
            the selected triplets are the hardest among the batch

"""


def euclidean_distance_matrix(x):
    """Efficient computation of Euclidean distance matrix
    Args:
      x: Input tensor of shape (batch_size, embedding_dim)

    Returns:
      Distance matrix of shape (batch_size, batch_size)
    """
    # step 1 - compute the dot product

    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())

    # step 2 - extract the squared Euclidean norm from the diagonal

    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)

    # step 3 - compute squared Euclidean distances

    # shape: (batch_size, batch_size)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

    # get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)

    # step 4 - compute the non-squared distances

    # handle numerical stability
    # derivative of the square root operation applied to 0 is infinite
    # we need to handle by setting any 0 to eps
    mask = (distance_matrix == 0.0).float()

    # use this mask to set indices with a value of 0 to eps
    distance_matrix += mask * eps

    # now it is safe to get the square root
    distance_matrix = torch.sqrt(distance_matrix)

    # undo the trick for numerical stability
    distance_matrix *= (1.0 - mask)

    return distance_matrix


def get_triplet_mask(labels):
    """compute a mask for valid triplets
    Args:
      labels: Batch of integer labels. shape: (batch_size,)
    Returns:
      Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
      A triplet is valid if:
      `labels[i] == labels[j] and labels[i] != labels[k]`
      and `i`, `j`, `k` are different.
    """
    # step 1 - get a mask for distinct indices

    # shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype = torch.bool, device = labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    # shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    # shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # step 2 - get a mask for valid anchor-positive-negative triplets

    # shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1)
    # shape: (batch_size, batch_size, batch_size)
    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # step 3 - combine two masks
    mask = torch.logical_and(distinct_indices, valid_indices)

    return mask


# Define the triplet loss function as provided
class BatchAllTripletLoss(nn.Module):
    """Uses all valid triplets to compute Triplet loss"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Compute distance matrix
        distance_matrix = euclidean_distance_matrix(embeddings)

        # Compute all triplet combinations
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        anchor_negative_dists = distance_matrix.unsqueeze(1)

        # Calculate triplet loss
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

        # Filter out invalid triplets
        mask = get_triplet_mask(labels)
        triplet_loss *= mask

        # Apply ReLU to filter out negative losses (easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # Calculate the number of positive losses
        num_positive_losses = (triplet_loss > 0).float().sum()

        # Average the positive losses
        triplet_loss = triplet_loss.sum() / (num_positive_losses + 1e-8)

        return triplet_loss
