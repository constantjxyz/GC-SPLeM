import torch
import torch.nn as nn
from model.attention import Attn_Net_Gated
import torch.nn.functional as F

class PathologyNet(torch.nn.Module):
    def __init__(self, params=dict()):
        super(PathologyNet, self).__init__()
        patho_embedding_dim = int(params['patho_embedding_dim']) # according to the input data, can be diff for diff datasets
        patho_dim_before_coattn = int(params['patho_dim_before_coattn'])  # need to be nailed
        patho_dropout_rate = float(params['patho_dropout_rate']) # need to be nailed

        fc = [
                nn.Linear(patho_embedding_dim, patho_dim_before_coattn), 
                nn.ELU(),
                nn.Dropout(patho_dropout_rate)]
        self.encoder = nn.Sequential(*fc)

            # CAN ADD Classifier

    def forward(self, x):
        features = self.encoder(x.float())

        # CAN ADD Classifier
        return features


# class PathologyMILNet(nn.Module):
#     '''
#     input: h_path_coattn, an embedding of pathology and gene, 
#             shape (gene_set_num, batch_size(1), embedding_coattn)
#     framework: path_transformer + global attention pooling
#     output:
#     '''
#     def __init__(self, params=dict()):
#         super(PathologyMILNet, self).__init__()
#         input_embed_dim = int(params['embedding_coattn_dim'])
#         nhead = int(params['pathology_mil_head'])
#         dropout_rate = float(params['pathology_mil_dropout_rate'])
#         mediate_dim = int(params['pathology_mil_mediate_dim'])
#         encoder_layer_num = int(params['pathology_mil_layer_num'])
#         path_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_embed_dim, nhead=nhead, dropout=dropout_rate, activation='relu',
#             dim_feedforward=mediate_dim)
#         self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=encoder_layer_num)
#         self.path_attention_head = Attn_Net_Gated(
#             L=input_embed_dim, D=input_embed_dim, dropout=dropout_rate, n_classes=1)
#         self.path_rho = nn.Sequential(
#             nn.Linear(input_embed_dim, input_embed_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
        
#     def forward(self, x):
#         pass
#         # input x: shape (gene_set_num, batch_size(1), input_embed_dim)
#         x = self.path_transformer(x)
#         # x shape: (gene_set_num, batch_size(1), input_embed_dim)
#         a, x = self.path_attention_head(x.squeeze())  
#         a = a.transpose(1, 0)
#         # x shape: (gene_set_num, input_embed_dim)
#         # a shape: (gene_set_num, 1)
#         x = torch.mm(F.softmax(a, dim=1), x)
#         # x shape: (1, input_embed_dim)
#         x = self.path_rho(x).squeeze()
#         # output x: shape (input_embed_dim), one dimension tensor
#         return x

# class PathologyNet(torch.nn.Module):
#     def __init__(self, **kwargs):
#         if kwargs['path_input'] == 'embedding':
#             self.net_name = 'pathology'
#             net_params = kwargs['']
#             fc = [
#                 nn.Linear(net_params[0], net_params[1]), 
#                 nn.ReLu(),
#                 nn.Dropout(net_params[2])]
#             self.encoder = nn.Sequential(*fc)

#             # CAN ADD Classifier
#         else:
#             raise NotImplementedError(kwargs['path_input'])

#     def forward(self, x):
#         features = self.encoder(x.float())

#         # CAN ADD Classifier
#         return features