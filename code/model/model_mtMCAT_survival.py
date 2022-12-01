import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pathology import PathologyNet
from model.mil import MILNet
from model.gene import GeneNet
from model.attention import CrossModalAttention
from model.fusion import FusionNet
from model.mt_survival import MTNet

class MTMCAT(nn.Module):
    def __init__(self, params=dict()):
        super(MTMCAT, self).__init__()

        self.path_net = PathologyNet(params=params) # already have
        self.gene_net = GeneNet(params=params) # already have

        self.coattn_net = CrossModalAttention(int(params['patho_dim_before_coattn']), 1 )

        self.path_mil_net = MILNet(mode='pathology_gene', params=params) # TODO add MIL module
        self.gene_mil_net = MILNet(mode='gene', params=params) # TODO add MIL module

        self.fusion_net = FusionNet(mode='concat_mlp', params=params) 

        self.mt_net = MTNet(params=params)
        self.device = params['device']

    def forward(self, data):
        # input shape:
            # pathology: shape (batch_size, patch_num, patho_embed_dim)
            # gene: [tensor shaped (batch_size, gene_num), ...]
            # clinical: [a, b, c, d, ...] length: (task_num)

        patho = data['pathology']
        gene = data['gene']
        clinical = data['clinical']

        patho = patho.to(self.device)
        gene = [sub_gene.to(self.device) for sub_gene in gene]
        clinical = [label.to(self.device) for label in clinical]

        h_path = self.path_net(patho)
        h_gene = self.gene_net(gene)
        
        # output shape:
            # h_path: shape (batch_size, patch_num, patho_embed_dim)
            # h_gene: [tensor shape (batch_size, gene_embed_dim), ...]

        h_path = h_path.transpose(1, 0)
        h_gene = torch.stack(h_gene)

        # input shape: h_path (patch_num, batch_size, patho_embedding)
        # gene: (gene_set_num, batch_size, gene_embedding)
        # h_path_gene, path_gene_coattn_weight = self.coattn_net(h_gene, h_path, h_path)
        h_path_gene, _ = self.coattn_net(h_gene, h_path, h_path)
        # output shape: 
            # h_path_gene: (gene_set_num, batch_size, gene_embedding)
            # path_gene_coattn_weight: (batch_size, gene_set_num, patch_num)
        
        # h_path_mil, gate_path_mil = self.path_mil_net(h_path_gene)
        # h_gene_mil, gate_gene_mil = self.gene_mil_net(h_gene)
        h_path_mil, _ = self.path_mil_net(h_path_gene)
        h_gene_mil, _ = self.gene_mil_net(h_gene)
        # output shape:
            # h_path_mil: shape(1, gene_embedding)
            # h_gene_mil: shape(1. gene_embedding)
        h = self.fusion_net(h_path_mil, h_gene_mil)
        # output shape: h shape(1. gene_embedding)
        # hazards, S, Y_hat, logits= self.mt_net(h.squeeze(0))
        hazards, S, Y_hat, _= self.mt_net(h.squeeze(0))
        # hazards: tensor shape: (1, n_class)
        # Y_hat: tensor shape: (1,1)
        # S: tensor shape:(1, n_class)
        c = clinical[0].clone().detach().to(self.device)
        label = clinical[2].clone().detach().to(self.device)
        event_time = clinical[1].clone().detach().to(self.device)
        # c: tensor shape (1)
        # label, event_time: tensor shape (1)
        return hazards, S, Y_hat, c, label, event_time

    def analyze_attention(self, data):
        # input shape:
            # pathology: shape (batch_size, patch_num, patho_embed_dim)
            # gene: [tensor shaped (batch_size, gene_num), ...]
            # clinical: [a, b, c, d, ...] length: (task_num)

        patho = data['pathology']
        gene = data['gene']
        clinical = data['clinical']
        

        patho = patho.to(self.device)
        gene = [sub_gene.to(self.device) for sub_gene in gene]
        clinical = [label.to(self.device) for label in clinical]

        h_path = self.path_net(patho)
        h_gene = self.gene_net(gene)
        
        # output shape:
            # h_path: shape (batch_size, patch_num, patho_embed_dim)
            # h_gene: [tensor shape (batch_size, gene_embed_dim), ...]

        h_path = h_path.transpose(1, 0)
        h_gene = torch.stack(h_gene)

        # input shape: h_path (patch_num, batch_size, patho_embedding)
        # gene: (gene_set_num, batch_size, gene_embedding)
        h_path_gene, path_gene_coattn_weight = self.coattn_net(h_gene, h_path, h_path)
        # output shape: 
            # h_path_gene: (gene_set_num, batch_size, gene_embedding)
            # path_gene_coattn_weight: (batch_size, gene_set_num, patch_num)
        h_path_mil, gate_path_mil = self.path_mil_net(h_path_gene)
        h_gene_mil, gate_gene_mil = self.gene_mil_net(h_gene)
        
        return path_gene_coattn_weight, gate_path_mil, gate_gene_mil

    def get_embedding(self, data):
        # input shape:
            # pathology: shape (batch_size, patch_num, patho_embed_dim)
            # gene: [tensor shaped (batch_size, gene_num), ...]
            # clinical: [a, b, c, d, ...] length: (task_num)

        patho = data['pathology']
        gene = data['gene']
        clinical = data['clinical']

        patho = patho.to(self.device)
        gene = [sub_gene.to(self.device) for sub_gene in gene]
        clinical = [label.to(self.device) for label in clinical]

        h_path = self.path_net(patho)
        h_gene = self.gene_net(gene)
        
        # output shape:
            # h_path: shape (batch_size, patch_num, patho_embed_dim)
            # h_gene: [tensor shape (batch_size, gene_embed_dim), ...]

        h_path = h_path.transpose(1, 0)
        h_gene = torch.stack(h_gene)

        # input shape: h_path (patch_num, batch_size, patho_embedding)
        # gene: (gene_set_num, batch_size, gene_embedding)
        h_path_gene, path_gene_coattn_weight = self.coattn_net(h_gene, h_path, h_path)
        # output shape: 
            # h_path_gene: (gene_set_num, batch_size, gene_embedding)
            # path_gene_coattn_weight: (batch_size, gene_set_num, patch_num)
        
        h_path_mil, gate_path_mil = self.path_mil_net(h_path_gene)
        h_gene_mil, gate_gene_mil = self.gene_mil_net(h_gene)
        # output shape:
            # h_path_mil: shape(1, gene_embedding)
            # h_gene_mil: shape(1. gene_embedding)
        h = self.fusion_net(h_path_mil, h_gene_mil)
        # output shape: h shape(1. gene_embedding)

        return h
        # hazards, S, Y_hat, logits= self.mt_net(h.squeeze(0))
        # # hazards: tensor shape: (1, n_class)
        # # Y_hat: tensor shape: (1,1)
        # # S: tensor shape:(1, n_class)
        # c = clinical[0].clone().detach().to(self.device)
        # label = clinical[2].clone().detach().to(self.device)
        # event_time = clinical[1].clone().detach().to(self.device)
        # # c: tensor shape (1)
        # # label, event_time: tensor shape (1)
        # return hazards, S, Y_hat, c, label, event_time
