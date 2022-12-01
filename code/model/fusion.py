'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-02-17 18:13:20
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-04 09:27:37
FilePath: /mtmcat/model/fusion.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
fusion for two one-dimensional embeddings shape like (256) 
'''

import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, mode='simple_concat', params=dict()):
        super(FusionNet, self).__init__()
        '''
        input: two embedding, h_gene_mil, h_patho_mil
        output: 
        '''
        self.mode = mode
        self.input_embed_dim = int(params['embedding_coattn_dim'])
        self.fusion_mediate_embed_dim = int(params['fusion_mediate_embed_dim'])
        self.fusion_dropout_rate = float(params['fusion_dropout_rate'])
        self.embedding_final = int(params['embedding_final'])
        self.mlp = nn.Sequential()
        if self.mode == 'concat_mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.input_embed_dim * 2, self.fusion_mediate_embed_dim), 
                nn.ReLU(),
                nn.Dropout(self.fusion_dropout_rate),
                nn.Linear(self.fusion_mediate_embed_dim, self.embedding_final),
                nn.ReLU(),
                nn.Dropout(self.fusion_dropout_rate),
            )
        

    def forward(self, h_patho_mil, h_gene_mil, ):
        if self.mode == 'simple_concat' or 'concat_mlp':
            h_mix = torch.cat([h_patho_mil, h_gene_mil], axis=-1)
            # h_mix: shape(batch_size, 2*input_embed_dim)
        if self.mode == 'concat_mlp':
            h_mix = self.mlp(h_mix)
            # h_mix: shape(batch_size, input_embed_dim)
        return h_mix

