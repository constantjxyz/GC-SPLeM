import torch
import torch.nn as nn

class GeneNet(torch.nn.Module):
    def __init__(self, params=dict()):
        super(GeneNet, self).__init__()
        # get parameters from params dict
        self.gene_set_dim = [int(dim) for dim in params['gene_set_dim']]
        self.gene_dim_before_coattn = int(params['gene_dim_before_coattn'])
        self.gene_dropout_rate = float(params['gene_dropout_rate'])
        # define module parts
        gene_set_networks = []
        for input_gene_set_dim in self.gene_set_dim:
            fc = [
                nn.Linear(input_gene_set_dim, self.gene_dim_before_coattn),
                nn.ELU(),
                nn.Dropout(self.gene_dropout_rate),
                nn.Linear(self.gene_dim_before_coattn, self.gene_dim_before_coattn),
                nn.ELU(),
                nn.Dropout(self.gene_dropout_rate),
            ]
            gene_set_networks.append(nn.Sequential(*fc))
        self.gene_set_networks = nn.ModuleList(gene_set_networks)

    def forward(self, x):
        pass
        '''
        input x: [tensor of set 1, tensor of set2, ...]
        length of set1, set2... : params['gene_set_dim']
        '''
        return [self.gene_set_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x)]

# class GeneNet(torch.nn.Module):
#     def __init__(self, params=dict()):
#         num_layers = 2
#         input_dim = int(params['input_dim'])
#         gene_dim = int(params['gene_dim'])
#         label_dim = int(params['label_dim'])
#         print('model initialize, now model parameters:', 'input dim:', params['input_dim'], 'number of layers:', 
#         params['num_layers'], 'device', params['device'])
#         super(GeneNet, self).__init__()
#         self.device = params['device']
#         hidden = [int(input_dim/5)]
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden[0]),
#             torch.nn.BatchNorm1d(hidden[0]),
#             torch.nn.ELU(),
#             # torch.nn.AlphaDropout(p=parameters.drop_out, inplace=False),
#         )
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Linear(hidden[0], gene_dim),
#             torch.nn.BatchNorm1d(gene_dim),
#             torch.nn.ELU(),
#             # torch.nn.AlphaDropout(p=parameters.drop_out, inplace=False),
#         )
#         self.encoder = torch.nn.Sequential(self.layer1, self.layer2)
#         self.classifier = torch.nn.Sequential(torch.nn.Linear(gene_dim, label_dim))
#         self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

#     def forward(self, x):
#         features = self.encoder(x.float())
#         output = self.classifier(features)
#         return output