'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-23 19:14:04
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-09-08 13:59:40
FilePath: /mtmcat/model/model_gnn_survival.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os
import torch
from torch import float32
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # self.dropout = nn.dropout(dropout=dropout_rate)
        # self.batch_norm = nn.BatchNorm1d(output_dim)
    def forward(self, features, adj):
        out = torch.mm(adj, features)
        out = self.linear(out)
        out = F.relu(out)
        # out = self.batch_norm(out)
        return out

class GCN_survival(nn.Module):
    def __init__(self, input_dim=64, hidden_dim1=32, hidden_dim2=16, params=dict()):
        super(GCN_survival, self).__init__()
        self.device = params['device']
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.gcn1 = GraphConvolution(64, 64)
        self.gcn2 = GraphConvolution(64, 64)
        self.classifier = nn.Linear(64, 4)
    
    def forward(self,features, adj):
        # input shape:
            # data: dictionary
            # adj, features: numpy array
            # clinical: [a, b, c, d, ...] length: (task_num)

        # features = data['features']
        # adj = data['adj']
        # clinical = data['clinical']

        features = features.to(self.device).float()
        adj = adj.to(self.device).float()
        # clinical = [label.to(self.device) for label in clinical]

        x = features
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
                
        all_logits = self.classifier(x)  # shape: (nodes_num, 4) or (nodes_num, 1)
        all_Y_hat = torch.topk(all_logits, 1, dim=-1)[1] # shape: (nodes_num,1)
        all_hazards = torch.sigmoid(all_logits) # shape: (nodes_num, 4)
        S = torch.cumprod(1 - all_hazards, dim=-1) # shape: (nodes_num, n_class)
        return (all_hazards, S, all_Y_hat, all_logits)