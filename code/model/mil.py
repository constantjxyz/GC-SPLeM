import torch
import torch.nn as nn
from model.attention import Attn_Net_Gated
import torch.nn.functional as F


class MILNet(nn.Module):
    '''
    input: h_path_coattn/h_gene, an embedding of pathology and gene, 
            shape (gene_set_num, batch_size(1), embedding_coattn)
    mode: pathology_gene or gene 
    framework: path_transformer + global attention pooling
    output:
    '''
    def __init__(self, mode = 'pathology', params=dict()):
        super(MILNet, self).__init__()
        input_embed_dim = int(params['embedding_coattn_dim'])
        self.mode = mode
        assert self.mode == 'pathology_gene' or 'gene'
        if self.mode == 'pathology_gene':
            nhead = int(params['pathology_mil_head'])
            dropout_rate = float(params['pathology_mil_dropout_rate'])
            mediate_dim = int(params['pathology_mil_mediate_dim'])
            encoder_layer_num = int(params['pathology_mil_layer_num'])
        elif self.mode == 'gene':
            nhead = int(params['gene_mil_head'])
            dropout_rate = float(params['gene_mil_dropout_rate'])
            mediate_dim = int(params['gene_mil_mediate_dim'])
            encoder_layer_num = int(params['gene_mil_layer_num']) 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_embed_dim, nhead=nhead, dropout=dropout_rate, activation='relu',
            dim_feedforward=mediate_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layer_num)
        self.attention_head = Attn_Net_Gated(
            L=input_embed_dim, D=input_embed_dim, dropout=dropout_rate, n_classes=1)
        self.rho = nn.Sequential(
            nn.Linear(input_embed_dim, input_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        pass
        # input x: shape (gene_set_num, batch_size(1), input_embed_dim)
        x = self.transformer(x)
        # x shape: (gene_set_num, batch_size(1), input_embed_dim)
        a, x = self.attention_head(x)  
        a = a.transpose(1, 0)
        a_raw = a
        
        # x shape: (gene_set_num, batch_size, input_embed_dim)
        # a shape: (batch_size, gene_set_num, 1)
        x = torch.matmul(F.softmax(a, dim=1).transpose(2, 1), x.transpose(1, 0))
        # x shape: (batch_size, 1, input_embed_dim)
        x = self.rho(x).squeeze(-2)
        # output x: shape (batch_size, input_embed_dim)
        return x, a_raw