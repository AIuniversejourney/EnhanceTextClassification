import torch
import torch.nn as nn
import dgl.function as fn

import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Sequential
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool


class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self, num_features, dim_h, num_classes, dropout=0.2):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(num_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
