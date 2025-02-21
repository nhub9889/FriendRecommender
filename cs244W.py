import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import dgl
import networkx as nx

USER = 0

# Đọc file edges và features
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

num_nodes = 0
feats = []


with open(file_feat) as f:
    for line in f:
        num_nodes += 1
        a = [int(x) for x in line.split()[1:]]
        feats.append(torch.tensor(a, dtype=torch.float))

feats = torch.stack(feats)


g = dgl.graph((edges_u, edges_v))
g.ndata['feat'] = feats


nx_g = g.to_networkx()


plt.figure(figsize=(10, 10))
nx.draw(nx_g, with_labels=True, node_color='lightblue', node_size=50, font_size=10, font_weight='bold')


plt.show()

