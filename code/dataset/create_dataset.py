'''
create the dataset for whole training process
'''
# import dependent modules
import os
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

class MtmcatDataset(Dataset):
    def __init__(self, patient_list, params=dict(), isTrain = False):
        super(PatientDataset).__init__()
        self.dataset_dir = params['dataset_dir']
        assert os.path.exists(self.dataset_dir)
        self.gene_dir = os.path.join(self.dataset_dir, 'gene.csv')
        self.pathology_dir = os.path.join(self.dataset_dir, 'pathology.csv') 
        self.wsi_feature_dir = params['wsi_feature_dir']
        self.clinical_dir = os.path.join(self.dataset_dir, 'clinical.csv')
        self.gene_set_dir = os.path.join(self.dataset_dir, 'gene_set.csv')
        self.patient_list = patient_list
        self.shuffle_mode = params['shuffle_mode']
        self.shuffle_type = params['shuffle_type']
        self.dataset_dir = params['dataset_dir']
        self.lack_patient = []
        if not isinstance(self.patient_list, list):
            self.patient_list = list(self.patient_list.patient_file['case_id'])
        self.gene_set_file = pd.read_csv(self.gene_set_dir)
        self.gene_set_num = int(params['gene_set_num'])
        assert self.gene_set_num == len(self.gene_set_file.columns)   # ensure the gene set
        # read the gene.csv, clinical.csv, pathology.csv in dataset folder
        self.clinical_file = pd.read_csv(self.clinical_dir)
        self.pathology_file = pd.read_csv(self.pathology_dir)
        self.gene_file = pd.read_csv(self.gene_dir)

        self.gene_set_names = []
        for gene_set_name in self.gene_set_file.columns:
            gene_names = []
            gene_set = self.gene_set_file[gene_set_name]
            for gene in gene_set:
                if type(gene) == str and gene in self.gene_file.columns:       # ensure that gene != Nan, cause length of different sets are different
                    gene_names.append(gene)
            self.gene_set_names.append(gene_names)

        self.clinical_columns_list = list(self.clinical_file.columns)   # return a list
        self.clinical_columns_list.remove('case_id')
        try:
            self.clinical_columns_list.remove('key_weight_label')
        except:
            print('this is ruijin. NO key_weight_label')
        self.task_num = len(self.clinical_columns_list) # how many tasks in the final model
        # self.clinical_class_num_list = [len(self.clinical_file[label].value_counts()) for label in self.clinical_columns_list]

        # #for all cases
        # self.random_feature_path = '/mnt/mnt-temp/TCGA/ruijin_random_patho/220818/'
        # self.pathology_file = self.pathology_file[self.pathology_file['case_id'].isin(self.patient_list)]
        # mean = np.load(os.path.join(self.dataset_dir, 'embedding/control', 'patho_mean.npy'))
        # std = np.load(os.path.join(self.dataset_dir, 'embedding/control', 'patho_std.npy'))
        # if True:
        #     if not os.path.exists(self.random_feature_path):
        #         os.makedirs(self.random_feature_path)
        #     for case_id in self.patient_list:
        #         slides_index = self.pathology_file[self.pathology_file['case_id'] == case_id].index
        #         slides_name = self.pathology_file.loc[slides_index, 'slide_id']
        #         patho_random_features_to_save = []
        #         for slide_name in slides_name:
        #             wsi_feature_path = os.path.join(self.wsi_feature_dir, 'pt_files', slide_name+'.pt')
        #             wsi_feature = torch.load(wsi_feature_path)
        #             patho_random_features_to_save.append(torch.from_numpy(np.random.normal(loc=mean, scale=std, size=wsi_feature.shape)))
        #         patho_random_features_to_save = torch.cat(patho_random_features_to_save, dim=0)
        #         torch.save(patho_random_features_to_save, os.path.join(self.random_feature_path, case_id + '.pt'))

        if isTrain: # Not used
            # print('Train params: ', params)

            if params['shuffle_scope'] == 'train':
                print('start shuffling')
                # only shuffle data in train dataset
                self.gene_file = self.gene_file[self.gene_file['case_id'].isin(self.patient_list)]
                self.pathology_file = self.pathology_file[self.pathology_file['case_id'].isin(self.patient_list)]
            if params['shuffle_type'] == 'gene':
                rand_num = int(params['rand_seed'])
                gene = self.gene_file.iloc[:, 1:]
                np.random.seed(rand_num)
                if params['shuffle_mode'] == 'random_50%':
                    lack_index = list(pd.Series(self.gene_file.index).sample(frac=0.5, random_state=rand_num))
                    self.lack_patient = [self.gene_file.loc[i, 'case_id'] for i in lack_index]
                    keep_index = np.setdiff1d(self.gene_file.index, lack_index) 
                    ref_file = self.gene_file.loc[keep_index, self.gene_file.columns[1:]]
                    if os.path.exists(os.path.join(params['dataset_dir'], 'embedding/control')):
                        mean = np.load(os.path.join(params['dataset_dir'], 'embedding/control', 'gene_mean.npy'))
                        std = np.load(os.path.join(params['dataset_dir'], 'embedding/control', 'gene_std.npy')) 
                    else:
                        mean, std = np.array(np.mean(ref_file)), np.array(np.std(ref_file))
                    self.gene_file.loc[lack_index, self.gene_file.columns[1:]] = np.random.normal(loc=mean, scale=std, size=(len(lack_index), len(self.gene_file.columns)-1))
                    pass
                else:
                    raise NotImplementedError(params['shuffle_mode'])
            elif params['shuffle_type'] == 'pathology':
                rand_num = int(params['rand_seed'])
                gene = self.gene_file.iloc[:, 1:]
                np.random.seed(rand_num)
                if params['shuffle_mode'] == 'pool_25%': 
                    self.lack_patient = [self.gene_file.loc[i, 'case_id'] for i in list(pd.Series(self.gene_file.index).sample(frac=0.25, random_state=rand_num))]
                elif params['shuffle_mode'] == 'pool_50%': 
                    self.lack_patient  = [self.gene_file.loc[i, 'case_id'] for i in list(pd.Series(self.gene_file.index).sample(frac=0.5, random_state=rand_num))]
                elif params['shuffle_mode'] == 'pool_75%': 
                    self.lack_patient  = [self.gene_file.loc[i, 'case_id'] for i in list(pd.Series(self.gene_file.index).sample(frac=0.75, random_state=rand_num))]
                elif params['shuffle_mode'] == 'random_50%': 
                    self.lack_patient  = [self.gene_file.loc[i, 'case_id'] for i in list(pd.Series(self.gene_file.index).sample(frac=0.5, random_state=rand_num))]
                    mean = np.load(os.path.join(self.dataset_dir, 'embedding/control', 'patho_mean.npy'))
                    std = np.load(os.path.join(self.dataset_dir, 'embedding/control', 'patho_std.npy'))
                    self.random_feature_path = '/mnt/mnt-temp/TCGA/ruijin_random_patho/220818/'        
                    # path to save random changed features
                    if False:
                        if not os.path.exists(self.random_feature_path):
                            os.makedirs(self.random_feature_path)
                        for case_id in self.gene_file['case_id'].values.tolist():
                            slides_index = self.pathology_file[self.pathology_file['case_id'] == case_id].index
                            slides_name = self.pathology_file.loc[slides_index, 'slide_id']
                            patho_random_features_to_save = []
                            for slide_name in slides_name:
                                wsi_feature_path = os.path.join(self.wsi_feature_dir, 'pt_files', slide_name+'.pt')
                                wsi_feature = torch.load(wsi_feature_path)
                                patho_random_features_to_save.append(torch.from_numpy(np.random.normal(loc=mean, scale=std, size=wsi_feature.shape)))
                            patho_random_features_to_save = torch.cat(patho_random_features_to_save, dim=0)
                            torch.save(patho_random_features_to_save, os.path.join(self.random_feature_path, case_id + '.pt'))
                else:
                    raise NotImplementedError(params['shuffle_mode'])

                 
            elif params['shuffle_type'] == 'False': 
                pass
            else:
                raise NotImplementedError(params['shuffle_type'])

    def __getitem__(self, index):
        '''
        input: index of case_id in the patient_list
        output: all data for a single patient, including
                1 pathology data matrix shaped (a,b), a as the number of patches across all wsi of a patient, 
                    b as the dim of patho-features
                m gene data matrix shaped (c), m as the number of gene set, c as the number of the gene in specific gene set
                a dictionary containing n labels according to clinical data
        '''
        case_id = self.patient_list[index]
        # get the pathology data according to case_id
        slides_index = self.pathology_file[self.pathology_file['case_id'] == case_id].index
        slides_name = self.pathology_file.loc[slides_index, 'slide_id']
        patho_features_list = []
        if self.shuffle_type == 'pathology' and case_id in self.lack_patient:
            if self.shuffle_mode == 'random_50%':
                patho_features_tensor = torch.load(os.path.join(self.random_feature_path, case_id + '.pt'))
            else:
                for slide_name in slides_name:
                    wsi_feature = torch.zeros(size=(10000,1024))
                    patho_features_list.append(wsi_feature)
                patho_features_tensor = torch.cat(patho_features_list, dim=0)
        else:
            for slide_name in slides_name:
                wsi_feature_path = os.path.join(self.wsi_feature_dir, 'pt_files', slide_name+'.pt')
                wsi_feature = torch.load(wsi_feature_path)
                patho_features_list.append(wsi_feature)
            patho_features_tensor = torch.cat(patho_features_list, dim=0)
                
        # get the gene data according to case_id
        gene_features_list = []
        for gene_names in self.gene_set_names:
            idx = self.gene_file[self.gene_file['case_id'] == case_id].index[0]
            value_set = self.gene_file.loc[idx, gene_names].values.reshape(-1).tolist()
            gene_features_list.append(torch.tensor(value_set).to(torch.float32))
        
        # get the clinical data according to case_id, return a list
        clinical_list = []
        idx = self.clinical_file[self.clinical_file['case_id'] == case_id].index[0]
        clinical_list = [self.clinical_file.loc[idx, column_name] for column_name in self.clinical_columns_list]

        # pack data for three modalities
        all_data = dict()
        all_data['case_id'] = case_id
        all_data['pathology'] = patho_features_tensor
        all_data['gene'] = gene_features_list
        all_data['all_gene'] = self.gene_file[self.gene_file['case_id'] == case_id].iloc[:, 1:].values
        all_data['clinical'] = clinical_list
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

