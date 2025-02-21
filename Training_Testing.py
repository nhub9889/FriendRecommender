import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import scipy.sparse as sp
import matplotlib.pyplot as plt
import dgl
import networkx as nx
from DotPredictor import GCN, GraphSAGE, DotPredictor
from sklearn.metrics import roc_auc_score

USER = 0

file_edges = f'facebook/{USER}.edges'
file_feat = f'facebook/{USER}.feat'

edges_u, edges_v = [], []

# Load edges file
with open(file_edges) as f:
    for line in f:
        e1, e2 = tuple(int(x) - 1 for x in line.split())
        edges_u.append(e1)
        edges_v.append(e2)

edges_u, edges_v = np.array(edges_u), np.array(edges_v)

# Load node features file
feats = []
with open(file_feat) as f:
    for line in f:
        a = [int(x) for x in line.split()[1:]]
        feats.append(torch.tensor(a, dtype=torch.float))

feats = torch.stack(feats)

# Construct graph
g = dgl.graph((edges_u, edges_v))  # Graph
g.ndata['feat'] = feats

# Split data into train and test
TEST_RATIO = 0.3
u, v = g.edges()

eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * TEST_RATIO)
train_size = g.number_of_edges() - test_size

# Get positive edges for train and test
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Negative edges
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

# Split negative edges for train and test
neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

# Create graph for positive and negative edges
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

# Remove edges from the graph and add self-loops
train_g = dgl.remove_edges(g, eids[:test_size])
train_g = dgl.add_self_loop(train_g)

def pipeline(model_name='GCN', hidden_size=16, out_size=16):
    if model_name == 'GCN':
        model = GCN(train_g.ndata['feat'].shape[1], hidden_size, out_size)
    elif model_name == 'SAGE':
        model = GraphSAGE(train_g.ndata['feat'].shape[1], hidden_size)

    pred = DotPredictor()

    def compute_loss(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_auc(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
        return roc_auc_score(labels, scores)

    # ----------- loss và optimizer -------------- #
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)


    all_logits = []
    for e in range(100):
        # forward
        h = model(train_g, train_g.ndata['feat'])  # lấy embeddings của các nút
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('Epoch {}, loss: {}'.format(e, loss))

    # ----------- kiểm tra kết quả ---------------- #
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc(pos_score, neg_score))

    # if model_name == "GCN":
    #     torch.save(model.state_dict(), 'gcn_model.pth')
    # else:
    #     torch.save(model.state_dict(), 'sage_model.pth')

    return h  # trả về node embeddings

def generate_rec(h, user_id=0):
    num_nodes = g.number_of_nodes()

    user_friends = set()
    for n1, n2 in zip(u, v):
        if int(n1) == user_id:
            user_friends.add(int(n2))
        if int(n2) == user_id:
            user_friends.add(int(n1))

    user_neg_u, user_neg_v = [], []
    for i in range(num_nodes):
        if i != user_id and i not in user_friends:
            user_neg_u.append(user_id)
            user_neg_v.append(i)

    user_g = dgl.graph((user_neg_u, user_neg_v), num_nodes=num_nodes)

    pred = DotPredictor()

    scores = []
    with torch.no_grad():
        for i, score in enumerate(pred(user_g, h)):
            scores.append((i, score.item()))

    scores.sort(key=lambda x: -x[1])
    recommendations = [{"id": score[0], "score": score[1]} for score in scores[:5]]

    return recommendations


h = pipeline('SAGE')