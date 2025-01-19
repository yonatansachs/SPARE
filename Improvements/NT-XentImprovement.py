""" using NT-Xent loss instead of original loss
also changed the graph build
to run this improvement, change the losses class to the folowing code
more chanes in line 68
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
   def __init__(self, temperature=0.5):
       super(NTXentLoss, self).__init__()
       self.temperature = temperature
       self.softmax = nn.Softmax(dim=-1)
       self.criterion = nn.CrossEntropyLoss(reduction='mean')
   def forward(self, anchor, positive):
       """
       Compute NT-Xent Loss between anchor and positive samples.
       Args:
           anchor (torch.Tensor): Anchor embeddings, shape (batch_size, dim)
           positive (torch.Tensor): Positive embeddings, shape (batch_size, dim)
       Returns:
           torch.Tensor: NT-Xent loss
       """
       # Normalize embeddings
       anchor = F.normalize(anchor, p=2, dim=1)
       positive = F.normalize(positive, p=2, dim=1)
       # Compute similarity matrix
       logits = torch.mm(anchor, positive.t()) / self.temperature
       # Labels are all diagonal elements (positive pairs)
       labels = torch.arange(anchor.size(0), device=anchor.device)
       loss = self.criterion(logits, labels)
       return loss
class SSLTaskWithNTXent(nn.Module):
   def __init__(self, opt):
       super(SSLTaskWithNTXent, self).__init__()
       self.nt_xent_loss = NTXentLoss(temperature=opt.temp)
       self.k = opt.k  # number of positive samples
   def forward(self, h_session, last_items_emb, pos_last_items_emb, neg_last_items_emb, pos_target_item_emb,
               neg_targets_item_emb):
       """
       Compute SSL loss using NT-Xent loss.
       Args:
           h_session: Session embeddings (batch_size, dim)
           last_items_emb: Last item embeddings (batch_size, dim)
           pos_last_items_emb: Positive last item embeddings (batch_size, k, dim)
           neg_last_items_emb: Negative last item embeddings (batch_size, k, dim)
           pos_target_item_emb: Positive target item embeddings (batch_size, dim)
           neg_targets_item_emb: Negative target item embeddings (batch_size, k, dim)
       """
       batch_size = h_session.size(0)
       # Create anchor embeddings
       anchor = h_session + last_items_emb  # (batch_size, dim)
       # Create positive embeddings by combining positive targets with their corresponding last items
       # Reshape pos_last_items_emb to match pos_target_item_emb
       pos_target_expanded = pos_target_item_emb.unsqueeze(1)  # (batch_size, 1, dim)
       positive = pos_target_expanded + pos_last_items_emb  # (batch_size, k, dim)
       # Reshape tensors for NT-Xent loss calculation
       anchor_repeated = anchor.unsqueeze(1).expand(-1, self.k, -1)  # (batch_size, k, dim)
       # Reshape to (batch_size * k, dim)
       anchor_flat = anchor_repeated.reshape(-1, anchor.size(-1))
       positive_flat = positive.reshape(-1, positive.size(-1))
       # Calculate NT-Xent loss
       loss = self.nt_xent_loss(anchor_flat, positive_flat)
       return loss

""" ***********
change the build_graph class to the following code 
more changes in line 200
**********"""
import argparse
import pickle
from collections import defaultdict
from math import sqrt

import numpy as np
import scipy
from scipy.sparse import diags
import time

from scipy.sparse.csgraph import dijkstra


# Build adjacency matrix with improvements
def build_adj(opt, num_node, verbose=True):
   # Load session data
   sessions = pickle.load(open(f'datasets/{opt.dataset}/all_train_seq.pkl', 'rb'))


   if verbose:
       print("Creating adjacency matrix")


   # Initialize adjacency matrix
   adj = np.zeros((num_node + 1, num_node + 1), dtype=int)


   # Fill adjacency matrix based on session data
   for sess in sessions:
       for i in range(len(sess) - 1):
           item_i = sess[i]
           item_j = sess[i + 1]
           adj[item_i, item_j] += 1


   if verbose:
       print("Max weight:", adj.max())


   # Filter out unreliable edges
   adj = np.where(adj <= opt.filter, 0, adj)


   # Add self-loops by adding identity matrix
   np.fill_diagonal(adj, adj.diagonal() + 1)


   if verbose:
       print("Number of non-zero edges (excluding diagonal):",
             np.count_nonzero(adj) - np.count_nonzero(adj.diagonal()))


   # Apply shortest paths using Dijkstra if necessary
   if opt.spare:
       rev_w_adj = adj.astype(np.float32)
       rev_mask = rev_w_adj.nonzero()
       max_v = max(rev_w_adj[rev_mask])
       rev_w_adj[rev_mask] = (max_v + 1) - rev_w_adj[rev_mask]


       if verbose:
           print("Finding shortest paths with lowest cost")
       adj = dijkstra(csgraph=rev_w_adj, directed=True, unweighted=False, limit=opt.limit)


       # Refill diagonal with original values
       np.fill_diagonal(adj, rev_w_adj.diagonal())
       adj[adj == np.inf] = 0


   # Sparse matrix creation and normalization
   adj_sparse = scipy.sparse.csr_matrix(adj)
   row_sum = np.array(adj.sum(axis=1))
   r_inv = np.power(row_sum, -0.5).flatten()
   r_mat_inv = diags(r_inv)
   adj_sparse = r_mat_inv.dot(adj_sparse).dot(r_mat_inv)


   if verbose:
       print("Adjacency matrix normalized")


   return adj, adj_sparse




def save_adj_matrix(adj_sparse, opt):
   """
   Save the sparse adjacency matrix to a file.
   """
   print("Writing adjacency matrix to file...")
   scipy.sparse.save_npz(f'datasets/{opt.dataset}/adj_global', adj_sparse)
   print("File saved successfully!")




if __name__ == '__main__':
   # Argument parsing for configurations
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', default='tmall', help='tmall/lastfm/retailrocket')
   parser.add_argument('--filter', type=int, default=0, help='Filter out unreliable edges below threshold.')
   parser.add_argument('--spare', type=int, default=1, help='Create adj based on shortest paths.')
   parser.add_argument('--limit', type=float, default=200, help='Max. search depth in Dijkstra.')
   opt = parser.parse_args()


   # Set number of nodes based on dataset
   if opt.dataset == 'tmall':
       opt.limit = 190  # max: 197
       num_node = 40727
   elif opt.dataset == 'retailrocket':
       opt.limit = 300  # max: 331
       num_node = 36968
   elif opt.dataset == 'lastfm':
       num_node = 38615
       opt.limit = 1526  # max: 1526


   # Build adjacency matrix
   print("Starting to build the adjacency matrix...")
   adj, adj_sparse = build_adj(opt, num_node)


   # Save the adjacency matrix to a file
   save_adj_matrix(adj_sparse, opt)


   print(adj_sparse.shape)
   print("Graph built successfully.")

"change line 28 in recommender to the following : "
self.ssl_task = SSLTaskWithNTXent(opt)

"change the sparsetosparse funciton in util to the following:"
def sparse2sparse(coo_matrix):
    # Ensure the matrix is in COO format
    if not isinstance(coo_matrix, scipy.sparse.coo_matrix):
        coo_matrix = coo_matrix.tocoo()

    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo_matrix.data)
    shape = coo_matrix.shape
    sparse_matrix = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return sparse_matrix

"""to correctly run the code, first run the build_graph file and then the main file after the graph is built"""