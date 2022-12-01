'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-22 14:13:51
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-07-24 17:26:26
FilePath: /mtmcat/dataset/create_gnn_dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
create the dataset for whole training process
'''
# import dependent modules
import os
from pyexpat import features
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import h5py

class PatientDataset(Dataset):
    def __init__(self, params=dict(), patient_keep='all', patient_keep_list = []):
        super(PatientDataset).__init__()
        self.dataset_dir = params['dataset_dir']
        assert os.path.exists(self.dataset_dir)
        self.patient_dir = os.path.join(self.dataset_dir, 'patient.csv')
        self.patient_file = pd.read_csv(self.patient_dir)
        # filter according given list
        if patient_keep != 'all':
            self.patient_file = self.patient_file.loc[patient_keep, :]
            self.patient_index = patient_keep
        else:
            if patient_keep_list != []:
                keep_index = []
                for case_id in patient_keep_list:
                    query = self.patient_file[self.patient_file['case_id'] == case_id]
                    if len(query) == 1:
                        idx = query.index[0]
                        keep_index.append(idx)
                self.patient_file = self.patient_file.loc[keep_index, :]
            self.patient_index = self.patient_file.index
        assert (self.patient_file.loc[self.patient_index, :] == self.patient_file).all().all()    # assert all the values in the self.patient_file what we get in __getitem__

    def __getitem__(self, index):
        idx = self.patient_index[index]
        return self.patient_file.loc[idx, 'case_id']
    
    def __len__(self):
        return len(self.patient_file)
    
    def return_patient_list(self):
        '''
        return the patients in self.patient_file as a list
        '''
        return list(self.patient_file['case_id'])
    
    def return_key_weight_label(self):
        '''return the key weight label as np.array'''
        # according to self.patient_file['key_weight_label']
        return np.array(self.patient_file['key_weight_label'])
    
    def return_patient_index(self):
        ''' return the patient index of self.patient_file in the patient.csv'''
        return list(self.patient_index)

class GNNDataset(Dataset):
    def __init__(self, patient_list, params=dict()):
        super(GNNDataset).__init__()
        self.dataset_dir = params['dataset_dir']
        self.clinical_dir = os.path.join(self.dataset_dir, 'clinical.csv')
        self.patient_list = patient_list
        self.clinical_file = pd.read_csv(self.clinical_dir)
        self.clinical_columns_list = list(self.clinical_file.columns)   # return a list
        self.clinical_columns_list.remove('case_id')
        assert len(np.unique(self.clinical_file['case_id'])) == len(self.clinical_file['case_id'])  #ensure no duplicate
        self.h5_dir = params['h5_dataset_dir']
        f = h5py.File(self.h5_dir, mode='r')
        self.features = np.array(f['embedding_matrix'], dtype=float)
        self.adj = np.array(f['normed_adj_matrix'], dtype=float)
        case_ids_h5 = np.array(f['nodes'], dtype=object)
        case_id_idx_dict = dict()
        for i in range(len(case_ids_h5)):
            case_id = case_ids_h5[i]
            case_id_idx_dict[case_id] = i
        f.close()
        self.case_id_idx_dict = case_id_idx_dict

    def __getitem__(self, index: int):
        case_id = self.patient_list[index]
        idx = self.clinical_file[self.clinical_file['case_id'] == case_id].index[0]
        clinical_list = [self.clinical_file.loc[idx, column_name] for column_name in self.clinical_columns_list]
        all_data = dict()
        # all_data['features'] = self.features
        # all_data['adj'] = self.adj
        all_data['clinical'] = clinical_list
        all_data['case_id'] = case_id
        all_data['case_id_idx'] = self.case_id_idx_dict[case_id]
        return all_data  

    def __len__(self):
        return len(self.patient_list)
    
    def return_key_weight_label(self):
        # according to self.clinical_file['key_weight_label']
        key_label_all = []
        for i in range(len(self.patient_list)):
            case_id = self.patient_list[i]
            idx = self.clinical_file[self.clinical_file['case_id'] == case_id].index[0]
            try:
                key_label = self.clinical_file.loc[idx, 'key_weight_label']
            except:
                key_label = self.clinical_file.loc[idx, 'label']
            key_label_all.append(key_label)
        return np.array(key_label_all)
    
    def return_features_adj(self):
        return self.features, self.adj

# class GNNDataset(Dataset):
#     def __init__(self, patient_list, params=dict()):
#         super(PatientDataset).__init__()
#         self.dataset_dir = params['dataset_dir']
#         assert os.path.exists(self.dataset_dir)
#         self.clinical_dir = os.path.join(self.dataset_dir, 'clinical.csv')
#         self.patient_list = patient_list
#         self.clinical_file = pd.read_csv(self.clinical_dir)
#         self.clinical_columns_list = list(self.clinical_file.columns)   # return a list
#         self.clinical_columns_list.remove('case_id')
#         assert len(np.unique(self.clinical_file['case_id'])) == len(self.clinical_file['case_id'])  #ensure no duplicate
#         self.patho_gene_embedding_dir = params['patho_gene_embedding_dir']
        
#     def __getitem__(self, index: int):
#         case_id = self.patient_list[index]
#         patho_gene_embedding_name = os.path.join(self.patho_gene_embedding_dir, case_id+'.pt')
#         patho_gene_embedding = torch.load(patho_gene_embedding_name).squeeze().to('cpu')

#         idx = self.clinical_file[self.clinical_file['case_id'] == case_id].index[0]
#         clinical_list = [self.clinical_file.loc[idx, column_name] for column_name in self.clinical_columns_list]
        
#         all_data = {}
#         all_data['patho_gene_embedding'] = patho_gene_embedding
#         all_data['clinical'] = clinical_list
#         all_data['case_id'] = case_id
#         return all_data

#     def __len__(self):
#         return len(self.patient_list)
    
#     def return_key_weight_label(self):
#         # according to self.clinical_file['key_weight_label']
#         key_label_all = []
#         for i in range(len(self.patient_list)):
#             case_id = self.patient_list[i]
#             idx = self.clinical_file[self.clinical_file['case_id'] == case_id].index[0]
#             try:
#                 key_label = self.clinical_file.loc[idx, 'key_weight_label']
#             except:
#                 key_label = self.clinical_file.loc[idx, 'label']
#             key_label_all.append(key_label)
#         return np.array(key_label_all)
    

