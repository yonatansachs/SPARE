""" Session based attention improvement (makes dynamic wheits based on item relationsships in sessions
 to run the improvement paste the following code instead of the recommender class
 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import GlobalItemConv
from model.losses import SSLTask
from model.losses import SSLTask




class SessionBasedAttention(nn.Module):
   def __init__(self, dim):
       super(SessionBasedAttention, self).__init__()
       self.query = nn.Linear(dim, dim)
       self.key = nn.Linear(dim, dim)
       self.value = nn.Linear(dim, dim)
       self.softmax = nn.Softmax(dim=-1)
       self.scale = dim ** 0.5


   def forward(self, session_emb, item_embs):
       """
       session_emb: Tensor of shape (batch_size, dim)
       item_embs: Tensor of shape (batch_size, seq_len, dim)
       """
       query = self.query(session_emb).unsqueeze(1)  # (batch_size, 1, dim)
       key = self.key(item_embs)  # (batch_size, seq_len, dim)
       value = self.value(item_embs)  # (batch_size, seq_len, dim)


       # Compute attention weights
       attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # (batch_size, 1, seq_len)
       attention_weights = self.softmax(attention_scores)  # (batch_size, 1, seq_len)


       # Compute weighted representation
       weighted_representation = torch.matmul(attention_weights, value)  # (batch_size, 1, dim)
       return weighted_representation.squeeze(1), attention_weights.squeeze(1)




class GraphRecommender(nn.Module):
   def __init__(self, opt, num_node, adj, len_session, n_train_sessions):
       super(GraphRecommender, self).__init__()
       self.opt = opt
       self.num_node = num_node
       self.len_session = len_session
       self.dim = opt.dim


       # Embedding layers
       self.item_embedding = nn.Embedding(num_node + 1, self.dim, padding_idx=0)
       self.pos_embedding = nn.Embedding(self.len_session, self.dim)


       # Graph Convolution
       self.item_conv = GlobalItemConv(layers=opt.layers)


       # Attention mechanism
       self.attention = SessionBasedAttention(dim=self.dim)


       # SSL Loss
       self.ssl_task = SSLTask(opt)


       # Dropout
       self.dropout = opt.dropout


       # Parameters for weighted L2 normalization
       self.w_k = opt.w_k


       # Adjacency matrix
       self.adj = adj


   def compute_sess_emb(self, inputs, seq_hidden):
       """
       Compute session embeddings using session-based attention.
       """
       # Mask for valid items
       mask = (inputs != 0).unsqueeze(-1)


       # Attention-based session embedding
       sess_emb, attention_weights = self.attention(seq_hidden.sum(dim=1), seq_hidden)
       return sess_emb


   def compute_con_loss(self, batch, sess_emb, item_embs):
       """
       Compute contrastive loss.
       """
       mask = (batch['inputs'] != 0).unsqueeze(-1)
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


       # Apply dropout
       hidden = F.dropout(hidden, self.dropout, training=self.training)


       alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
       seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)


       # Compute session embeddings
       sess_emb = self.compute_sess_emb(inputs, seq_hidden)


       # Weighted L2 normalization
       select = self.w_k * F.normalize(sess_emb, dim=-1, p=2)
       graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)


       # Compute scores
       scores = torch.matmul(select, graph_item_embs_norm.transpose(1, 0))


       # Compute contrastive loss if required
       con_loss = torch.Tensor(0)
       if cl:
           con_loss = self.compute_con_loss(batch, select, graph_item_embs_norm)


       return scores, con_loss

