"""
change the build graph code to the following code, then run the build graph class to build the new graph. then continue to line 100 for the next changes
"""
import pickle
import argparse
import numpy as np
import scipy.sparse
from tqdm import tqdm


def build_dynamic_weight_adj(opt, num_node, verbose=True):
    """
    Builds an adjacency matrix with dynamic weights based on session interactions.
    Args:
        opt: Command-line options containing dataset and configuration parameters.
        num_node: Total number of nodes/items in the dataset.
        verbose: Whether to print progress and debugging information.

    Returns:
        adj: Dense adjacency matrix with dynamic weights.
        adj_sparse: Symmetrical normalized sparse adjacency matrix.
    """
    # Load session data
    sessions = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.pkl', 'rb'))

    if verbose: print("Creating adjacency matrix with dynamic weights...")

    # Initialize adjacency matrix
    adj = np.zeros((num_node + 1, num_node + 1), dtype=float)

    # Compute edge weights dynamically based on co-occurrence frequency in sessions
    for sess in tqdm(sessions, desc="Processing sessions"):
        for i in range(len(sess) - 1):
            item_i = sess[i]
            item_j = sess[i + 1]
            adj[item_i, item_j] += 1
            adj[item_j, item_i] += 1  # Make graph undirected for dynamic weight propagation

    if verbose: print(f"Max weight before normalization: {adj.max()}")

    # Apply edge filtering to remove weak connections
    adj = np.where(adj <= opt.filter, 0, adj)

    # Transform weights dynamically (e.g., logarithmic scaling)
    adj = np.log1p(adj)  # Logarithmic transformation reduces the impact of high-frequency edges

    # Add self-loops to ensure proper node connectivity
    np.fill_diagonal(adj, adj.diagonal() + 1)

    if verbose:
        print(f"Non-zero edges after filtering: {np.count_nonzero(adj) - np.count_nonzero(adj.diagonal())}")

    # Normalize the adjacency matrix symmetrically (GCN normalization)
    if verbose: print("Normalizing adjacency matrix...")
    rowsum = np.array(adj.sum(axis=1))  # Sum of each row
    r_inv = np.power(rowsum, -0.5, where=rowsum > 0).flatten()  # D^(-1/2)
    r_mat_inv = scipy.sparse.diags(r_inv)  # Diagonal matrix with inverse square root of degrees
    adj_sparse = r_mat_inv.dot(scipy.sparse.csr_matrix(adj)).dot(r_mat_inv)  # Symmetric normalization

    # Convert to sparse COO format for efficient storage
    adj_sparse = adj_sparse.tocoo()

    return adj, adj_sparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build adjacency matrix with dynamic weights for session-based recommendation")
    parser.add_argument('--dataset', default='tmall', help='Dataset name (e.g., tmall, lastfm, retailrocket)')
    parser.add_argument('--filter', type=int, default=0,
                        help='Filter out unreliable edges below the specified threshold.')
    opt = parser.parse_args()

    # Define the number of nodes/items for each dataset
    num_node = 0
    if opt.dataset == 'tmall':
        num_node = 40727
    elif opt.dataset == 'retailrocket':
        num_node = 36968
    elif opt.dataset == 'lastfm':
        num_node = 38615
    else:
        raise ValueError("Invalid dataset specified. Please choose from: tmall, retailrocket, lastfm.")

    print("Configuration:", opt)
    print(f"Building adjacency matrix for dataset: {opt.dataset}...")

    # Build the adjacency matrix
    adj, adj_sparse = build_dynamic_weight_adj(opt, num_node)

    # Save the sparse adjacency matrix to a file
    output_path = f'datasets/{opt.dataset}/adj_dynamic.npz'
    print(f"Saving adjacency matrix to {output_path}...")
    scipy.sparse.save_npz(output_path, adj_sparse)

    print("Adjacency matrix shape:", adj_sparse.shape)
    print("Graph construction completed successfully.")



"""
******************************
change the layers class to the following code
******************************

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalItemConv(nn.Module):
    def __init__(
            self,
            spare=True,
            layers=1,
            feat_drop=0.0
    ):
        super(GlobalItemConv, self).__init__()
        self.spare = spare
        self.layers = layers
        self.feat_drop = nn.Dropout(feat_drop)

    def forward(self, x, adj):
        h = x
        final = [x]
        for i in range(self.layers):
            # Adjust adjacency matrix for dynamic weights
            if self.spare:
                weighted_adj = adj.coalesce()  # Ensure sparsity is maintained
                h = torch.sparse.mm(weighted_adj, h)
            else:
                h = torch.mm(adj, h)  # Dense matrix multiplication if not sparse

            h = F.normalize(h, dim=-1, p=2)
            h = self.feat_drop(h)
            final.append(h)
        if self.layers > 1:
            h = torch.sum(torch.stack(final), dim=0) / (self.layers + 1)
        return h

    def __repr__(self):
        return '{}(n_layers={},dropout={})'.format(self.__class__.__name__, self.layers, self.feat_drop)