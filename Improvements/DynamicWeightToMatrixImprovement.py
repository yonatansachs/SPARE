""" this is the dynamic weights addition to the adjacency matrix improvement attempt.
To run this improvement, paste the following code instead of the recommender class and change a single line in the main file.
change line 93 in the original main file to
model = trans_to_cuda(GraphRecommenderWithEnhancedGraph(opt, num_node, sparse_global_adj, len_session=train_data.max_len,
                                           n_train_sessions=len(train_data)))
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


from model.layers import GlobalItemConv
from model.losses import SSLTask




class TransformerEncoderLayer(nn.Module):
   def __init__(self, dim, num_heads, dropout=0.1):
       super(TransformerEncoderLayer, self).__init__()
       self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
       self.norm1 = nn.LayerNorm(dim)
       self.norm2 = nn.LayerNorm(dim)
       self.ff = nn.Sequential(
           nn.Linear(dim, dim * 4),
           nn.ReLU(),
           nn.Linear(dim * 4, dim)
       )
       self.dropout = nn.Dropout(dropout)


   def forward(self, x):
       # Self-attention layer
       attn_output, _ = self.attn(x, x, x)
       x = self.norm1(x + self.dropout(attn_output))
       # Feed-forward layer
       ff_output = self.ff(x)
       x = self.norm2(x + self.dropout(ff_output))
       return x


class GraphRecommenderWithEnhancedGraph(nn.Module):
   def __init__(self, opt, num_node, adj, len_session, n_train_sessions):
       super(GraphRecommenderWithEnhancedGraph, self).__init__()
       self.opt = opt
       self.num_node = num_node
       self.len_session = len_session
       self.dim = opt.dim


       self.item_embedding = nn.Embedding(num_node + 1, self.dim, padding_idx=0)
       self.pos_embedding = nn.Embedding(self.len_session, self.dim)


       self.ssl_task = SSLTask(opt)
       self.item_conv = GlobalItemConv(layers=opt.layers)


       self.adj = self._enhance_adj(adj)
       self.dropout = opt.dropout


       # Position-wise attention parameters
       self.w_k = opt.w_k


   def _enhance_adj(self, adj):
       """Enhance adjacency matrix by adding weights dynamically."""
       # Convert adjacency matrix to dense format for manipulation
       adj_dense = adj.to_dense()


       # Add dynamic weights based on node degrees
       degrees = torch.sum(adj_dense, dim=1, keepdim=True)
       enhanced_adj = adj_dense / degrees


       # Reconvert to sparse format
       return enhanced_adj.to_sparse()


   def compute_sess_emb(self, item_seq, hidden):
       batch_size = hidden.shape[0]
       mask = (item_seq != 0).unsqueeze(-1)


       # Session Embedding using mean pooling
       sess_emb = torch.sum(hidden * mask, dim=1) / torch.sum(mask, dim=1)
       return sess_emb


   def compute_con_loss(self, batch, sess_emb, item_embs):
       mask = torch.unsqueeze((batch['inputs'] != 0), -1)
       last_item_pos = torch.sum(mask, dim=1) - 1
       last_items = torch.gather(batch['inputs'], dim=1, index=last_item_pos).squeeze()
       last_items_emb = item_embs[last_items]


       pos_last_items_emb = item_embs[batch['pos_last_items']]
       neg_last_items_emb = item_embs[batch['neg_last_items']]


       pos_target_item_emb = item_embs[batch['targets']]
       neg_targets_item_emb = item_embs[batch['neg_targets']]


       con_loss = self.ssl_task(sess_emb, last_items_emb, pos_last_items_emb, neg_last_items_emb,
                                pos_target_item_emb, neg_targets_item_emb)


       return con_loss


   def forward(self, batch, cl=False):
       items, inputs, alias_inputs = batch['items'], batch['inputs'], batch['alias_inputs']
       graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)
       hidden = graph_item_embs[items]


       # Dropout
       hidden = F.dropout(hidden, self.dropout, training=self.training)


       alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
       seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)


       # Compute session embeddings
       sess_emb = self.compute_sess_emb(inputs, seq_hidden)


       # Weighted L2 normalization
       select = self.w_k * F.normalize(sess_emb, dim=-1, p=2)
       graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)


       scores = torch.matmul(select, graph_item_embs_norm.transpose(1, 0))


       con_loss = torch.Tensor(0)
       if cl:
           con_loss = self.compute_con_loss(batch, select, graph_item_embs_norm)


       return scores, con_loss
