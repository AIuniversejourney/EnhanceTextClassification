import torch as th
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

from model.torch_gat import GAT
from model.torch_gcn import GCN
from model.torch_graphsage import GraphSAGE
from model.torch_gin  import GIN


class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model, nb_class):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model, nb_class, m, gcn_layers,
                 n_hidden, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.training = True
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers - 1,
            activation=F.relu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred


class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model, nb_class, m, gcn_layers, heads,
                 n_hidden, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.training = True
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
            num_layers=gcn_layers - 1,
            in_dim=self.feat_dim,
            num_hidden=n_hidden,
            num_classes=nb_class,
            heads=[heads] * (gcn_layers - 1) + [1],
            activation=F.elu,
            feat_drop=dropout,
            attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred



class BertGraphSAGE(th.nn.Module):
    def __init__(self, pretrained_model, nb_class, m, gcn_layers, n_hidden, dropout=0.5):
        super(BertGraphSAGE, self).__init__()
        self.m = m
        self.training = True
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GraphSAGE(
            num_layers=gcn_layers,
            in_dim=self.feat_dim,
            num_hidden=n_hidden,
            num_classes=nb_class,
            aggregator_type='mean',
            activation=F.relu,
            feat_drop=dropout,
            negative_slope=0.2,
            residual=False
        )
        self.W = th.nn.Linear(self.feat_dim + n_hidden, n_hidden)

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = F.softmax(cls_logit, dim=1)
        gcn_feats = self.gcn(g.ndata['cls_feats'], g)
        combined_feats = th.cat([cls_feats, gcn_feats], dim=1)
        sage_feats = F.relu(self.W(combined_feats))
        sage_logit = th.mean(sage_feats, dim=0)
        sage_pred = F.softmax(sage_logit, dim=1)
        pred = (sage_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GINConv(nn.Module):
#     def __init__(self, out_dim, aggr='add', dropout=0.2, bias=True):
#         super(GINConv, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(out_dim, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#         )
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_dim))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.mlp.apply(self._reset)
#         if self.bias is not None:
#             self.bias.data.fill_(0)
#
#     def forward(self, x, edge_index):
#         out = self.propagate(edge_index, x=x)
#         out = self.mlp(out)
#         if self.bias is not None:
#             out = out + self.bias
#         return out
#
#     def message(self, x_j):
#         return x_j
#
#     def update(self, aggr_out, x):
#         out = aggr_out + x
#         return out
#
#     def _reset(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0.0)
#         elif isinstance(m, nn.BatchNorm1d):
#             nn.init.constant_(m.weight.data, 1.0)
#             nn.init.constant_(m.bias.data, 0.0)
