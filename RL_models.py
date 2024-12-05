import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from graphsage.neigh_samplers import MLNeighborSampler  
import tensorflow as tf

class RL_GCN(nn.Module):
    """
    GCN model with RL-based neighbor sampling.
    """

    def __init__(self, num_features, num_classes, sampler, num_hops=2, hidden_size=64):
        super(RL_GCN, self).__init__()
        self.num_hops = num_hops
        self.sampler = sampler
        self.hidden_size = hidden_size

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_size))
        for _ in range(num_hops - 1):
            self.convs.append(SAGEConv(hidden_size, hidden_size))

        self.lin = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index, batch):
    # Call the sampler with the correct arguments
        sampled_neighbors = self.sampler(x, edge_index=edge_index, batch=batch, num_samples=self.hidden_size)

        # Unpack all the outputs from the sampler
        adj_lists, att_lists, adj_lists_numnz, edge_index, edge_weight = sampled_neighbors

        # Use the adjacency lists (or edge_index/edge_weight if needed) in the GCN layers
        for conv in self.convs:
        #    x = F.relu(conv(x, edge_index))
            x = F.relu(conv(x, edge_index))

        # Pass through the final linear layer
        x = self.lin(x)
        return x


    def loss(self, predictions, targets):
        """
        Compute the loss for the model.
        :param predictions: Model predictions
        :param targets: Ground-truth labels
        :return: Cross-entropy loss
        """
        return F.cross_entropy(predictions, targets)


class RL_Sampler:
    """
    RL-based node sampler integrated with the GCN model.
    Works with sparse adjacency matrices.
    """

    def __init__(self, adj_matrix, features, hidden_size):
        if not isinstance(adj_matrix, tf.sparse.SparseTensor):
            # Convert sparse matrix to TensorFlow SparseTensor
            adj_matrix = tf.sparse.SparseTensor(
                indices=np.array(adj_matrix.nonzero()).T,
                values=adj_matrix.data,
                dense_shape=adj_matrix.shape
            )
        self.adj_matrix = adj_matrix
        self.features = features
        self.hidden_size = hidden_size
        self.sampler = MLNeighborSampler(adj_matrix, features)

    def sample(self, node_ids, num_samples):
        """
        Perform sampling for given node IDs.
        :param node_ids: IDs of the nodes to sample neighbors for
        :param num_samples: Number of neighbors to sample per node
        :return: Sampled neighbors and related outputs
        """
        return self.sampler(node_ids, num_samples=num_samples)

    def __call__(self, x, edge_index, batch, num_samples):
        """
        Make the sampler callable for integration with GCN models.
        :param x: Node features (unused here)
        :param edge_index: Edge index (unused here, adjacency is already stored)
        :param batch: Batch of node IDs for sampling
        :param num_samples: Number of neighbors to sample per node
        :return: Outputs from the sampler
        """
        return self.sample(batch, num_samples)