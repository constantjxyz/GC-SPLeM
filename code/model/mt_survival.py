'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-03-14 16:04:23
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-04 09:28:21
FilePath: /mtmcat/model/mt_survival.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
classifier for specific tasks
'''

from types import new_class
import torch
import torch.nn as nn

class MTNet(nn.Module):
    def __init__(self, params=dict()):
        super(MTNet, self).__init__()
        '''
        input: h_mix (input_embed_dim), one-dimensional embedding
        '''
        self.input_embed_dim = int(params['embedding_final'])
        self.n_class = int(params['sample_class_num'])
        self.classifier = nn.Linear(self.input_embed_dim, self.n_class)

    def forward(self, x):
        logits = self.classifier(x).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=-1)[1] # shape: (1,1)
        hazards = torch.sigmoid(logits) # shape: (1, n_class)
        S = torch.cumprod(1 - hazards, dim=-1) # shape: (1, n_class)
        return (hazards, S, Y_hat, logits)

# params = {
#     'embedding_coattn_dim':256,
#     'sample_class_num':4
# }
# model = MTNet(params = params)
# input = torch.rand(size=(256,))
# hazards, S, Y_hat = model.forward(input)
# print('ok')  
# print('end')    