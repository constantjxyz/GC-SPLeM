'''
run train, test and cross validation process
'''
# import dependent modules
import enum
import os
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
import pandas as pd
from collections import Counter

# import dependent functions from run_train_test.py and utils.py
from engine.utils import save_split
from dataset.create_gnn_dataset import PatientDataset, GNNDataset
from engine.train_test_gnn import train_survival, test_survival

# define cross_validation
def run_train_test(train_patient, val_patient, test_patient, params=dict()):
    # training by perslide or by person
    train_tensor_dataset = GNNDataset(train_patient, params=params)
    val_tensor_dataset = GNNDataset(val_patient, params=params)
    test_tensor_dataset = GNNDataset(test_patient, params=params)

    print('Train params: ', params)
    print('-'*80)

    # running different train and test process according to specific task
    task = params['task']
    assert task in ['survival', 'stage', 'path_stage', 'path_survival']
    if task == 'survival':
        # run survival test task
        model, best_val_loss = train_survival(train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, params=params)
        _, _, _ = test_survival(test_tensor_dataset, model, mode='test', params=params)
    # elif task == 'stage':
    #     # run classification task by using MCAT model (pathology data and gene data)
    #     model, best_val_loss = train(train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, params=params)
    #     _, _  = test(test_tensor_dataset, model, mode='test', params=params)
    # elif task == 'path_stage':
    #     # run classification task by using CLAM model (only pathology data)
    #     model, best_val_loss = train_path(train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, params=params)
    #     _, _, = test_path(test_tensor_dataset, model, mode='test', params=params)
    # elif task == 'path_survival':
    #     # run survival test task
    #     model, best_val_loss = train_path_survival(train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, params=params)
    #     _, _, _ = test_path_survival(test_tensor_dataset, model, mode='test', params=params)

    return best_val_loss

def run_cross_validation(train_dataset, test_dataset, params=dict()):
    folds = eval(params['cross_validation_folds'])
    random_state = int(params['rand_seed'])
    stratify_split = params['stratify']
    if folds > 1:  
        # need to run cross validation 
        assert stratify_split in ['True', 'False']
        if stratify_split == 'True':
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        else:
            skf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        val_loss_all = []
        train_val_index = train_dataset.return_patient_index()
        key_weight_label = train_dataset.return_key_weight_label()
        for fold, (train, val) in enumerate(skf.split(train_val_index, key_weight_label)):
            print('')
            print('-'*80)
            print(f'run cross validation fold {fold+1}, / {folds}all')
            train_patient_index= [train_val_index[i] for i in train]
            val_patient_index= [train_val_index[i] for i in val]
            train_patient = PatientDataset(params=params, patient_keep=train_patient_index)
            val_patient = PatientDataset(params=params, patient_keep=val_patient_index)

            #save the split of dataset
            if params['save_split_dir'] != 'False':
                assert os.path.exists(params['save_split_dir'])
                save_split(train_patient, val_patient, test_dataset, save_dir=os.path.join(params['save_split_dir'],params['ckpt']+'cv'+fold+'.csv'))



            tmp_params = params.copy()
            tmp_params['ckpt'] += '%d' % fold
            val_loss = run_train_test(train_patient, val_patient, test_dataset, params=tmp_params)
            val_loss_all.append(val_loss)
    else:  
        # no need to run cross validation
        val_loss_all = []
        train_val_index = train_dataset.return_patient_index()
        key_weight_label = train_dataset.return_key_weight_label()
        assert stratify_split in ['True', 'False']
        if stratify_split == 'True':
            key_weight_label = key_weight_label
            train, val = train_test_split(train_val_index, test_size=0.25, shuffle=True, random_state=random_state, stratify=key_weight_label)
        else:
           train, val = train_test_split(train_val_index, test_size=0.25, shuffle=True, random_state=random_state)

        print('')
        print('-'*80)
        train_patient = PatientDataset(params=params, patient_keep=train)
        val_patient = PatientDataset(params=params, patient_keep=val)

        # split the dataset according to csv file
        if str(params['split_file']) != 'nan':
            if os.path.exists(params['split_file']):
                split_file = pd.read_csv(params['split_file'])
                train_patient_list = list(split_file['train'].dropna())
                val_patient_list = list(split_file['validation'].dropna())
                test_patient_list = list(split_file['test'].dropna())
                train_patient = PatientDataset(params=params, patient_keep_list=train_patient_list)
                val_patient = PatientDataset(params=params, patient_keep_list=val_patient_list)
                test_dataset = PatientDataset(params=params, patient_keep_list=test_patient_list)
        #save the split of dataset
        if params['save_split_dir'] != 'False':
            assert os.path.exists(params['save_split_dir'])
            save_split(train_patient, val_patient, test_dataset, save_dir=os.path.join(params['save_split_dir'],params['ckpt']+'.csv'))

        val_loss = run_train_test(train_patient, val_patient, test_dataset, params=params)
        val_loss_all.append(val_loss)
    val_loss_average = np.mean(np.array(val_loss_all))
    print('the average validation loss across all folds: ', val_loss_average)