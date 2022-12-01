import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Attn_Net_Gated
from model.pathology import PathologyNet
from model.mil import MILNet
from model.gene import GeneNet
from model.attention import CrossModalAttention
from model.fusion import FusionNet
from model.mt_survival import MTNet

class CLAM_path(nn.Module):
    def __init__(self, params=dict()):
        super(CLAM_path, self).__init__()

        self.path_net = PathologyNet(params=params) # already have

        self.path_mil_net = Attn_Net_Gated(L= int(params['embedding_coattn_dim']), D = int(params['embedding_coattn_dim']),\
            dropout= float(params['pathology_mil_dropout_rate']), n_classes = 1)

        # self.fusion_net = FusionNet(mode='concat_mlp', params=params) 

        self.mt_net = MTNet(params=params)
        self.device = params['device']

        instance_classifiers = [nn.Linear(int(params['embedding_coattn_dim']), 2) for i in range(int(params['task_num'] * 2))]
        instance_classifiers = [nn.Linear(int(params['embedding_coattn_dim']), 2) for i in range(int(1 * 2))]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.subtyping = False

        k_sample=8
        self.k_sample = 8

        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        instance_loss_fn = instance_loss_fn.cuda(self.device)
        self.instance_loss_fn = instance_loss_fn

        self.instance_eval = False
        if params['use_inst'] == 'True':
            self.instance_eval = True
            self.n_classes = 2
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, data):
        # input shape:
            # pathology: shape (batch_size, patch_num, patho_embed_dim)
            # gene: [tensor shaped (batch_size, gene_num), ...]
            # clinical: [a, b, c, d, ...] length: (task_num)

        patho = data['pathology'][0]
        # gene = data['gene']
        clinical = data['clinical']

        patho = patho.to(self.device)
        # gene = [sub_gene.to(self.device) for sub_gene in gene]
        clinical = [label.to(self.device) for label in clinical]


        h_path = self.path_net(patho)
        # h_gene = self.gene_net(gene)
        
        # # output shape:
        #     # h_path: shape (batch_size, patch_num, patho_embed_dim)
        #     # h_gene: [tensor shape (batch_size, gene_embed_dim), ...]

        # h_path = h_path.transpose(1, 0)
        # h_gene = torch.stack(h_gene)

        # # input shape: h_path (patch_num, batch_size, patho_embedding)
        # # gene: (gene_set_num, batch_size, gene_embedding)
        # h_path_gene, path_gene_coattn_weight = self.coattn_net(h_gene, h_path, h_path)
        # # output shape: 
        #     # h_path_gene: (gene_set_num, batch_size, gene_embedding)
        #     # path_gene_coattn_weight: (batch_size, gene_set_num, patch_num)
        
        A, h_path_mil = self.path_mil_net(h_path.squeeze())

        A = torch.transpose(A, 0, 1)
        A = F.softmax(A, dim=1)
        # h_gene_mil = self.gene_mil_net(h_gene)
        # # output shape:
        #     # h_path_mil: shape(gene_embedding)
        #     # h_gene_mil: shape(gene_embedding)
        # h = self.fusion_net(h_path_mil, h_gene_mil)

        if self.instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(clinical[0].long(), num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h_path_mil, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h_path_mil, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h_path_mil)

        # print(M.shape)
        hazards, S, Y_hat, logits= self.mt_net(M.squeeze())
        # hazards: tensor shape: (1, n_class)
        # Y_hat: tensor shape: (1,1)
        # S: tensor shape:(1, n_class)
        c = clinical[0].clone().detach().to(self.device)
        label = clinical[2].clone().detach().to(self.device)
        event_time = clinical[1].clone().detach().to(self.device)
        # c: tensor shape (1)
        # label, event_time: tensor shape (1)
        if self.instance_eval:
            return hazards, S, Y_hat, c, label, event_time, total_inst_loss
        else:
            return hazards, S, Y_hat, c, label, event_time,

