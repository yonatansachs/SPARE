"""
To run this improvement, paste this code instead of the recommender file
need to change some more files, see line 130 of this code
******************************************
******************************************
IMPORTANT! the project runs from the main file not the files we are changing! This is true for all improvements
******************************************
******************************************
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from model.losses import SSLTask
from model.layers import GlobalItemConv

class GraphRecommender(nn.Module):
    def __init__(self, opt, num_node, adj, len_session, n_train_sessions, feature_dim):
        super(GraphRecommender, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.len_session = len_session

        self.dim = opt.dim
        self.feature_dim = feature_dim

        self.item_embedding = nn.Embedding(num_node + 1, self.dim, padding_idx=0)
        self.feature_embedding = nn.Linear(self.feature_dim, self.dim)  # Embed content features

        self.ssl_task = SSLTask(opt)

        self.item_conv = GlobalItemConv(layers=opt.layers)
        self.w_k = opt.w_k
        self.adj = adj
        self.dropout = opt.dropout

        self.n_sessions = n_train_sessions
        self.memory_bank = torch.empty((n_train_sessions, self.dim))

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_sess_emb(self, item_seq, hidden, rev_pos=True, attn=True):
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]
        mask = torch.unsqueeze((item_seq != 0), -1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, seq_len, 1)
        nh = hidden

        if rev_pos:
            pos_emb = self.item_embedding.weight[:seq_len]
            pos_emb = torch.flip(pos_emb, [0])  # reverse order
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
            nh = torch.tanh(nh)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        if attn:
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            sess_emb = torch.sum(beta * hidden, 1)
        else:
            sess_emb = torch.sum(nh * hidden, 1)

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

    def forward(self, batch, item_features, cl=False):
        items, inputs, alias_inputs = batch['items'], batch['inputs'], batch['alias_inputs']

        # Compute graph item embeddings
        graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)

        # Select features for the batch items
        feature_embs = item_features[items]

        # Adjust dimensions of feature_embs to match graph_item_embs
        feature_embs = F.linear(feature_embs, torch.eye(self.dim, feature_embs.shape[-1]))

        # Combine embeddings
        combined_embs = graph_item_embs[items] + feature_embs

        hidden = combined_embs
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

        sess_emb = self.compute_sess_emb(inputs, seq_hidden, rev_pos=True, attn=True)

        select = self.w_k * F.normalize(sess_emb, dim=-1, p=2)
        graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)

        scores = torch.matmul(select, graph_item_embs_norm.transpose(1, 0))

        con_loss = torch.Tensor(0)
        if cl:
            con_loss = self.compute_con_loss(batch, select, graph_item_embs_norm)

        return scores, con_loss




""" *******************************
***********************************
Change trainer to the following code
need to change some more, see line 320
***********************************
***********************************"""
import time
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

import numpy as np

from model.losses import *
from util import *


def trans_to_cuda(variable):
    return variable  # Always use CPU


def trans_to_cpu(variable):
    return variable  # Always use CPU


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
            item_features = torch.zeros(model.num_node + 1, model.feature_dim)

            scores, _ = model(batch, item_features=item_features)

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

    def get_item_features(self):
        # Create features for all items in the graph
        return torch.zeros(self.model.num_node + 1, self.model.feature_dim)

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
                item_features = self.get_item_features()  # Generate item features

                self.optimizer.zero_grad()

                scores, con_loss = self.model(batch, item_features=item_features, cl=self.contrastive)

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


""" *******************************
***********************************
Change main to the following code
***********************************
***********************************"""

import argparse

import scipy.sparse
import torch

from util import *
import pickle

from model.recommender import *
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='lastfm/tmall/retailrocket')
parser.add_argument('--len-session', type=int, default=50, help='maximal session length')

parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--layers', type=int, default=1, help='the number of gnn layers')
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')

parser.add_argument('--cl', type=int, default=1, help='Use contrastive base loss with pos and neg sessions')
parser.add_argument('--k', type=int, default=4, help='Sample size for pos and neg contrastive samples')
parser.add_argument('--temp', type=float, default=0.2, help='Temperature parameter for CL')
parser.add_argument('--sim', type=str, default='bleu',
                    help='Similarity measure for sessions: bleu/jaccard/cosine/levenshtein')
parser.add_argument('--beta', type=float, default=0.05, help='weighting contrastive loss')
parser.add_argument('--w-k', type=int, default=12, help='weight l2 normalization, ~10-20')

parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--log-interval', type=int, default=500, help='print the loss after this number of iterations')
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--seed', type=int, default=2022)

opt = parser.parse_args()

def main():
    seed = init_seed(opt.seed)
    print(seed)

    if opt.dataset == 'retailrocket':
        num_node = 36968
        opt.w_k = 12
        opt.dropout = 0.2
        opt.beta = 0.2
        opt.k = 16
        opt.temp = 0.2
    elif opt.dataset == 'tmall':
        num_node = 40727
        opt.w_k = 16
        opt.dropout = 0.4
        opt.beta = 0.2
        opt.k = 8
        opt.temp = 0.2
    elif opt.dataset == 'lastfm':
        num_node = 38615
        opt.w_k = 17
        opt.dropout = 0.4
        opt.beta = 0.05
        opt.k = 4
        opt.temp = 0.7
        opt.layers = 2

    feature_dim = 16  # Define the feature dimension for content-based features

    print(opt)
    print('reading dataset')

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.pkl', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.pkl', 'rb'))

    global_adj_coo = scipy.sparse.load_npz('datasets/' + opt.dataset + '/adj_global.npz')
    sparse_global_adj = trans_to_cuda(sparse2sparse(global_adj_coo))

    train_data = DataSampler(opt, train_data, opt.len_session, num_node, train=True)
    test_data = DataSampler(opt, test_data, opt.len_session, num_node, train=False)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                              shuffle=False, pin_memory=False)
    model = trans_to_cuda(GraphRecommender(opt, num_node, sparse_global_adj, len_session=train_data.max_len,
                                           n_train_sessions=len(train_data), feature_dim=feature_dim))
    print(model)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        opt=opt,
        Ks=[5, 10, 20],
    )

    print('start training')
    best_results = trainer.train(opt.epochs, opt.log_interval)

    print('\n')
    print(opt)

if __name__ == '__main__':
    main()
