import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.utils import from_scipy_sparse_matrix
from RL_models import RL_GCN, RL_Sampler
from scipy.sparse import load_npz

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cell_line', default='E122', type=str)
parser.add_argument('-rf', '--regression_flag', default=1, type=int)
parser.add_argument('-me', '--max_epoch', default=1000, type=int)
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
parser.add_argument('-es', '--embed_layer_size', default=5, type=int)
parser.add_argument('-cn', '--num_graph_conv_layers', default=2, type=int)
parser.add_argument('-gs', '--graph_conv_layer_size', default=256, type=int)
parser.add_argument('-ln', '--num_lin_layers', default=3, type=int)
parser.add_argument('-ls', '--lin_hidden_layer_size', default=256, type=int)
parser.add_argument('-rs', '--random_seed', default=0, type=int)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for L2 regularization.')
parser.add_argument('--max_degree', type=int, default=128, help='Maximum degree for neighbor sampling.')

args = parser.parse_args()

# Set Random Seed
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

# Paths
base_path = os.getcwd()
data_dir = os.path.join(base_path, 'data', args.cell_line)
adj_matrix_path = os.path.join(data_dir, 'hic_sparse.npz')
features_path = os.path.join(data_dir, 'np_hmods_norm_chip_10000bp.npy')
labels_path = os.path.join(data_dir, f'np_nodes_lab_genes_reg{args.regression_flag}.npy')

# Load Data
adj_matrix = load_npz(adj_matrix_path)
features = np.load(features_path)[:, 1:]  # Ignore node IDs
labels = np.load(labels_path)

# Prepare allLabs for labels
labeled_indices = labels[:, 0].astype(int)
allLabs = -1 * np.ones(features.shape[0])  # Default -1 for unlabeled nodes
geneLabs = labels[:, 1]
allLabs[labeled_indices] = geneLabs
y = torch.tensor(allLabs, dtype=torch.long if args.regression_flag == 0 else torch.float)

# Create targetNode_mask
targetNode_mask = torch.tensor(labeled_indices, dtype=torch.long)

# Filter adjacency matrix and features for labeled nodes
adj_matrix_labeled = adj_matrix[labeled_indices, :][:, labeled_indices]
edges, edge_weights = from_scipy_sparse_matrix(adj_matrix_labeled)

x = torch.tensor(features, dtype=torch.float)

# Split data into train/val/test
num_labeled_nodes = len(labeled_indices)
pred_idx_shuff = torch.randperm(num_labeled_nodes)
train_split = int(0.7 * num_labeled_nodes)
val_split = int(0.85 * num_labeled_nodes)
train_idx = pred_idx_shuff[:train_split]
val_idx = pred_idx_shuff[train_split:val_split]
test_idx = pred_idx_shuff[val_split:]

# Define Masked Loss Function
def masked_loss(predictions, targets):
    # Only consider valid (non -1) targets
    valid_mask = targets != -1
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    return torch.nn.functional.cross_entropy(predictions, targets)




num_classes = (y[y != -1].max().item() + 1) if args.regression_flag == 0 else 1
hidden_size = args.graph_conv_layer_size

print(f"Features shape: {x.shape}")
print(f"Number of classes: {num_classes}")
print(f"Hidden size: {hidden_size}")

# Initialize model
sampler = RL_Sampler(adj_matrix_labeled, features, hidden_size=64)
model = RL_GCN(num_features=x.shape[1], num_classes=num_classes, sampler=sampler, hidden_size=hidden_size)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Training Loop
for epoch in range(args.max_epoch):
    model.train()
    optimizer.zero_grad()

    # Get model predictions
    predictions = model(x, edges, batch=train_idx)

    # Filter predictions to include only those corresponding to train_idx
    predictions = predictions[train_idx] 

    print(f"Epoch {epoch + 1}")
    print(f"Predictions shape: {predictions.shape}")  # Debugging
    print(f"Train labels shape: {y[train_idx].shape}")  # Debugging
    loss = masked_loss(predictions, y[train_idx])
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(x, edges, batch=val_idx)
        val_loss = masked_loss(val_predictions, y[val_idx])

    print(f"Epoch {epoch + 1}/{args.max_epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Save Model
output_dir = os.path.join(data_dir, 'saved_runs')
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, f"RL_GCN_{args.cell_line}.pth"))
