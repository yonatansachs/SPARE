"""
to run this improvement first place the following code instead of the build graph and run the class
"""
import pickle
import argparse
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra
import time


def build_adj(opt, num_node, verbose=True):
    # Load sessions
    sessions = pickle.load(open(f'datasets/{opt.dataset}/all_train_seq.pkl', 'rb'))

    if verbose: print("Creating sequential adjacency matrix")
    adj = np.zeros((num_node + 1, num_node + 1), dtype=int)

    # Build adjacency matrix
    for sess in sessions:
        for i in range(len(sess) - 1):
            item_i = sess[i]
            item_j = sess[i + 1]
            adj[item_i, item_j] += 1

    if verbose: print(f"Max weight before filtering: {adj.max()}")

    # Filter unreliable edges
    adj = np.where(adj <= opt.filter, 0, adj)

    # Add self-loops
    np.fill_diagonal(adj, adj.diagonal() + 1)

    if verbose: print(f"Non-zero entries before dynamic weighting: {np.count_nonzero(adj)}")

    if opt.spare:
        # Reverse edge weights for shortest path calculation
        rev_w_adj = adj.astype(np.float32)
        rev_mask = rev_w_adj.nonzero()
        max_v = rev_w_adj[rev_mask].max()

        rev_w_adj[rev_mask] = (max_v + 1) - rev_w_adj[rev_mask]

        if verbose: print("Finding shortest paths with reversed weights")
        adj = dijkstra(csgraph=rev_w_adj, directed=True, unweighted=False, limit=opt.limit)

        # Revert weights
        np.fill_diagonal(adj, rev_w_adj.diagonal())
        adj[adj == np.inf] = 0
        mask = adj.nonzero()
        max_v = adj[mask].max()
        adj[mask] = (max_v + 1) - adj[mask]

    if verbose: print(f"Non-zero entries after dynamic weighting: {np.count_nonzero(adj)}")

    if verbose: print("Normalizing adjacency matrix")
    adj_sparse = scipy.sparse.csr_matrix(adj)

    # Normalize the adjacency matrix
    rowsum = np.array(adj_sparse.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    adj_sparse = d_mat_inv_sqrt @ adj_sparse @ d_mat_inv_sqrt

    adj_sparse = adj_sparse.tocoo()
    return adj, adj_sparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tmall', help='Dataset: tmall/lastfm/retailrocket')
    parser.add_argument('--filter', type=int, default=0, help='Filter out unreliable edges below threshold.')
    parser.add_argument('--spare', type=int, default=1, help='Create adjacency matrix based on shortest paths.')
    parser.add_argument('--limit', type=float, default=200, help='Max search depth in Dijkstra.')
    opt = parser.parse_args()

    # Dataset-specific settings
    if opt.dataset == 'tmall':
        opt.limit = 190  # max: 197
        num_node = 40727
    elif opt.dataset == 'retailrocket':
        opt.limit = 300  # max: 331
        num_node = 36968
    elif opt.dataset == 'lastfm':
        opt.limit = 1526  # max: 1526
        num_node = 38615
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")

    print(opt)
    adj, adj_sparse = build_adj(opt, num_node)

    print("Saving adjacency matrix to file...")
    scipy.sparse.save_npz(f'datasets/{opt.dataset}/adj_global.npz', adj_sparse)

    print(f"Adjacency matrix shape: {adj_sparse.shape}")
    print("Graph construction completed.")





"""
next,  place this code instead of util
"""





import numpy as np
import time
import random
import scipy
import torch
from torch.utils.data import Dataset
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from model.distances import jaccard, tanimoto, cosine, dameraulevenshtein


def sparse2sparse(coo_matrix):
    v1 = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = coo_matrix.shape
    return torch.sparse.LongTensor(i, v, torch.Size(shape))


def dense2sparse(matrix):
    a_ = scipy.sparse.coo_matrix(matrix)
    v1 = a_.data
    indices = np.vstack((a_.row, a_.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = a_.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def handle_data(inputs, train_len=None):
    len_data = [len(nowData) for nowData in inputs]
    max_len = max(len_data) + 1 if train_len is None else min(max(len_data), train_len) + 1
    us_pois = [upois + [0] * (max_len - le) if le < max_len else upois[-max_len:] for upois, le in zip(inputs, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len for le in len_data]
    return us_pois, us_msks, max_len


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def init_seed(seed=None):
    if seed is None or seed == 0:
        seed = int(time.time() * 1000 // 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return seed


class DataSampler(Dataset):
    def __init__(self, opt, sessions, max_len, num_node, train=True):
        self.opt = opt
        inputs, mask, len_max = handle_data(sessions[0], max_len)
        self.sessions = sessions[0]
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(sessions[1])
        self.mask = np.asarray(mask)
        self.length = len(sessions[0])
        self.max_len = len_max
        self.num_node = num_node
        self.vn_id = num_node + 1
        self.train = train
        self.k = opt.k
        self.num_subsample = self.k * 2
        self.similarity = opt.sim
        self.item_session_id_map = {}
        self.target_session_id_map = {}
        for sess_idx, sess in enumerate(sessions[0]):
            for item_id in sess:
                self.item_session_id_map.setdefault(item_id, set()).add(sess_idx)
        for sess_idx, target in enumerate(sessions[1]):
            self.target_session_id_map.setdefault(target, []).append(sess_idx)

    def get_data(self, idx):
        u_input, mask, target = self.inputs[idx], self.mask[idx], self.targets[idx]
        node = np.unique(u_input)
        items = node.tolist() + [0] * (self.max_len - len(node))
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        return u_input, mask, target, items, alias_inputs

    def get_pos_last_items(self, session_id):
        target = self.targets[session_id]
        target_sessions_ids = list(self.target_session_id_map[target])
        if len(target_sessions_ids) > 1:
            target_sessions_ids.remove(session_id)
        pos_ids = random.choices(target_sessions_ids, k=self.k)
        return [self.inputs[p_id][sum(self.mask[p_id]) - 1] for p_id in pos_ids], pos_ids

    def session_similarity(self, session, pos_sessions, neg_sessions):
        sessions = [session] + pos_sessions
        if self.similarity == 'jaccard':
            return np.array([[jaccard(set(sess), set(neg_sess)) for neg_sess in neg_sessions] for sess in sessions]).sum(axis=0)
        elif self.similarity == 'tanimoto':
            return np.array([[tanimoto(set(sess), set(neg_sess)) for neg_sess in neg_sessions] for sess in sessions]).sum(axis=0)
        elif self.similarity == 'cosine':
            return np.array([[cosine(set(sess), set(neg_sess)) for neg_sess in neg_sessions] for sess in sessions]).sum(axis=0)
        elif self.similarity == 'levenshtein':
            return np.array([[-dameraulevenshtein.damerau_levenshtein_distance(sess, neg_sess) for neg_sess in neg_sessions] for sess in sessions]).sum(axis=0)
        elif self.similarity == 'bleu':
            return [bleu_score.sentence_bleu(sessions, neg_sess, smoothing_function=SmoothingFunction().method7, weights=[0.5, 0.3, 0.15, 0.05]) for neg_sess in neg_sessions]

    def get_neg_sessions(self, session_id, pos_ids, pos_last_items):
        session = self.sessions[session_id]
        pos_sessions = [list(x) for x in set(tuple(x) for x in [self.sessions[pos_id] for pos_id in pos_ids])]
        candidate_neg_ids = {idx for item_id in session for idx in self.item_session_id_map.get(item_id, []) if self.targets[idx] != self.targets[session_id]}
        candidate_neg_ids -= {idx for idx in candidate_neg_ids if set(self.sessions[idx]).issubset(set(session))}
        candidate_neg_ids -= {idx for idx in candidate_neg_ids if self.sessions[idx][-1] in pos_last_items + [session[-1]]}
        if len(candidate_neg_ids) < self.k:
            neg_ids = list(candidate_neg_ids) + random.choices(range(len(self.sessions)), k=self.k - len(candidate_neg_ids))
        else:
            neg_sessions = [self.sessions[idx] for idx in candidate_neg_ids]
            sim = self.session_similarity(session, pos_sessions, neg_sessions)
            neg_ids = [list(candidate_neg_ids)[i] for i in np.argsort(sim)[::-1][:self.k]]
        return [self.inputs[n_id][sum(self.mask[n_id]) - 1] for n_id in neg_ids], [self.targets[n_id] for n_id in neg_ids]

    def __getitem__(self, index):
        u_input, mask, target, items, alias_inputs = self.get_data(index)
        pos_last_items, neg_last_items, neg_targets = None, None, None
        if self.train and self.opt.cl:
            pos_last_items, pos_ids = self.get_pos_last_items(index)
            neg_last_items, neg_targets = self.get_neg_sessions(index, pos_ids, pos_last_items)
        return {
            "alias_inputs": torch.tensor(alias_inputs),
            "items": torch.tensor(items),
            "mask": torch.tensor(mask),
            "targets": torch.tensor(target),
            "inputs": torch.tensor(u_input),
            "index": torch.tensor(index),
            "pos_last_items": torch.tensor(pos_last_items) if pos_last_items else torch.tensor([]),
            "neg_last_items": torch.tensor(neg_last_items) if neg_last_items else torch.tensor([]),
            "neg_targets": torch.tensor(neg_targets) if neg_targets else torch.tensor([]),
        }

    def __len__(self):
        return self.length





"""
 next, place the following code instead of recommender
"""




from functools import reduce
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from model.layers import *
from model.losses import *


class GraphRecommender(nn.Module):
    def __init__(self, opt, num_node, adj, len_session, n_train_sessions):
        super(GraphRecommender, self).__init__()
        self.opt = opt

        # Initialize dimensions and configurations
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.len_session = len_session
        self.dim = opt.dim
        self.w_k = opt.w_k
        self.adj = adj
        self.dropout = opt.dropout
        self.n_sessions = n_train_sessions

        # Embedding layers for items and session positions
        self.item_embedding = nn.Embedding(num_node + 1, self.dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.len_session, self.dim)

        # Self-supervised learning task
        self.ssl_task = SSLTask(opt)

        # Graph convolution for global item embedding
        self.item_conv = GlobalItemConv(layers=opt.layers)

        # Memory bank for sessions
        self.memory_bank = torch.empty((n_train_sessions, self.dim))

        # Positional attention weights and GLU (Gated Linear Unit)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters."""
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_sess_emb(self, item_seq, hidden, rev_pos=True, attn=True):
        """Compute session embeddings with optional reverse positional attention."""
        batch_size, seq_len, _ = hidden.size()
        mask = (item_seq != 0).unsqueeze(-1)  # Mask for valid items

        # Compute average session embedding
        hs = torch.sum(hidden * mask, dim=1) / torch.sum(mask, dim=1)
        hs = hs.unsqueeze(1).repeat(1, seq_len, 1)

        nh = hidden

        # Apply reverse positional embedding
        if rev_pos:
            pos_emb = self.pos_embedding.weight[:seq_len]
            pos_emb = torch.flip(pos_emb, [0]).unsqueeze(0).repeat(batch_size, 1, 1)
            nh = torch.tanh(torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1))

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        # Attention mechanism for session embedding
        if attn:
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            sess_emb = torch.sum(beta * hidden, dim=1)
        else:
            sess_emb = torch.sum(nh * hidden, dim=1)

        return sess_emb

    def compute_con_loss(self, batch, sess_emb, item_embs):
        """Compute contrastive loss for self-supervised learning."""
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
        """Forward pass for the Graph Recommender."""
        items, inputs, alias_inputs = batch['items'], batch['inputs'], batch['alias_inputs']

        # Global item embeddings
        graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)
        hidden = graph_item_embs[items]

        # Apply dropout
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        # Session-specific embeddings
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

        # Reverse position attention
        sess_emb = self.compute_sess_emb(inputs, seq_hidden, rev_pos=True, attn=True)

        # Weighted L2 normalization for recommendation scoring
        select = self.w_k * F.normalize(sess_emb, dim=-1, p=2)
        graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)
        scores = torch.matmul(select, graph_item_embs_norm.transpose(1, 0))

        # Contrastive loss computation
        con_loss = torch.Tensor([0])
        if cl:
            con_loss = self.compute_con_loss(batch, select, graph_item_embs_norm)

        return scores, con_loss





"""
next, place the following code instead of trainer
"""





import time
from collections import defaultdict
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from model.losses import *
from util import *

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def prepare_batch(batch):
    batch_dict = {
        "alias_inputs": trans_to_cuda(batch["alias_inputs"]).long(),
        "items": trans_to_cuda(batch["items"]).long(),
        "mask": trans_to_cuda(batch["mask"]).long(),
        "targets": trans_to_cuda(batch["targets"]).long(),
        "inputs": trans_to_cuda(batch["inputs"]).long(),
        "index": trans_to_cuda(batch["index"]).long(),
        "pos_last_items": trans_to_cuda(batch["pos_last_items"]).long(),
        "neg_last_items": trans_to_cuda(batch["neg_last_items"]).long(),
        "neg_targets": trans_to_cuda(batch["neg_targets"]).long(),
    }
    return batch_dict

def evaluate(model, data_loader, Ks=[10, 20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)
    with torch.no_grad():
        for batch in data_loader:
            batch = prepare_batch(batch)

            scores, _ = model(batch)

            loss = F.cross_entropy(scores, batch['targets'])
            results['Loss'] -= loss.item()

            batch_size = scores.size(0)
            num_samples += batch_size
            topk = torch.topk(scores, k=max_K, sorted=True)[1]
            targets = batch['targets'].unsqueeze(-1)
            for K in Ks:
                hit_ranks = torch.where(topk[:, :K] == targets)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += torch.log2(1 + hit_ranks).reciprocal().sum().item()
    for metric in results:
        results[metric] /= num_samples
    return results

def print_results(results, epochs=None):
    print('Metric\t' + '\t'.join(results.keys()))
    print(
        'Value\t' +
        '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()])
    )
    if epochs is not None:
        print('Epoch\t' + '\t'.join([str(epochs[metric]) for metric in results]))

class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            opt,
            Ks=[10, 20]
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.batch = 0
        self.patience = opt.patience
        self.Ks = Ks
        self.contrastive = opt.cl
        self.beta = opt.beta

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def train(self, epochs, log_interval=100):
        max_results = defaultdict(float)
        max_results['Loss'] = -np.inf
        max_epochs = defaultdict(int)
        bad_counter = 0
        t = time.time()
        total_loss, total_con_loss, mean_loss, mean_con_loss = 0, 0, 0, 0
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                batch = prepare_batch(batch)

                self.optimizer.zero_grad()

                scores, con_loss = self.model(batch, cl=self.contrastive)

                loss = self.loss_function(scores, batch['targets'])

                if self.contrastive:
                    combined_loss = loss + self.beta * con_loss
                    combined_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()

                if log_interval:
                    mean_loss += loss.item() / log_interval
                    mean_con_loss += torch.mean(con_loss).item() / log_interval

                total_loss += loss.item()
                total_con_loss += torch.mean(con_loss).item()

                if log_interval and self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Con-Loss = {mean_con_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss, mean_con_loss = 0, 0
                self.batch += 1

            curr_results = evaluate(
                self.model, self.test_loader, Ks=self.Ks
            )

            if log_interval:
                print(f'\nEpoch {self.epoch}:')
                print('Loss:\t%.3f' % total_loss)
                print('Con-Loss:\t%.3f' % total_con_loss)
                print_results(curr_results)

            any_better_result = False
            for metric in curr_results:
                if curr_results[metric] > max_results[metric]:
                    max_results[metric] = curr_results[metric]
                    max_epochs[metric] = self.epoch
                    any_better_result = True

            if any_better_result:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == self.patience:
                    break

            self.scheduler.step()
            self.epoch += 1
            total_loss = 0.0
            total_con_loss = 0.0

        print('\nBest results')
        print_results(max_results, max_epochs)
        return max_results






"""
next, place the following code instead of layers
"""





import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalItemConv(nn.Module):
    def __init__(
            self,
            spare=True,
            layers=1,
            feat_drop=0.0,
            dynamic_weights=False
    ):
        """
        Initialize the GlobalItemConv module.

        Parameters:
        - spare (bool): Use sparse matrix operations.
        - layers (int): Number of layers for convolution.
        - feat_drop (float): Dropout rate for features.
        - dynamic_weights (bool): Enable dynamic weight adjustments.
        """
        super(GlobalItemConv, self).__init__()
        self.spare = spare
        self.layers = layers
        self.feat_drop = nn.Dropout(feat_drop)
        self.dynamic_weights = dynamic_weights

    def compute_dynamic_weights(self, adj):
        """
        Compute dynamic weights for the adjacency matrix.

        Parameters:
        - adj (torch.sparse.FloatTensor): Sparse adjacency matrix.

        Returns:
        - torch.sparse.FloatTensor: Adjusted adjacency matrix with dynamic weights.
        """
        with torch.no_grad():
            weights = adj.coalesce().values()
            max_weight = weights.max()
            dynamic_weights = max_weight - weights + 1  # Reverse weights
            adjusted_adj = torch.sparse.FloatTensor(adj.indices(), dynamic_weights, adj.size())
            return adjusted_adj

    def forward(self, x, adj):
        """
        Forward pass of the module.

        Parameters:
        - x (torch.Tensor): Input feature matrix.
        - adj (torch.sparse.FloatTensor): Sparse adjacency matrix.

        Returns:
        - torch.Tensor: Output feature matrix after convolution.
        """
        h = x
        final = [x]

        # Adjust adjacency matrix weights dynamically if enabled
        if self.dynamic_weights:
            adj = self.compute_dynamic_weights(adj)

        for i in range(self.layers):
            h = torch.sparse.mm(adj, h)
            h = F.normalize(h, dim=-1, p=2)
            h = self.feat_drop(h)
            final.append(h)

        if self.layers > 1:
            h = torch.sum(torch.stack(final), dim=0) / (self.layers + 1)

        return h

    def __repr__(self):
        """
        Representation of the module.
        """
        return '{}(n_layers={}, dropout={}, dynamic_weights={})'.format(
            self.__class__.__name__, self.layers, self.feat_drop.p, self.dynamic_weights
        )

"""
now run the main file
"""

