
# import dependent modules
import torch
import numpy as np
import pandas as pd
import os
import wandb
from collections import Counter
import time
import argparse
import h5py
import shutil

# import dependent functions
from engine.utils import save_model, load_model
from setting.get_parameter import get_param_dict
from dataset.create_dataset import PatientDataset,PerslideDataset, MtmcatDataset
from model.model_mtMCAT_survival import MTMCAT as MTMCAT_survival
from model.model_mtMCAT import MTMCAT

# set parameters from csv
parser = argparse.ArgumentParser()
parser.add_argument('--setting_dir', type=str, default='./setting/gnn/ruijin_63/get_embedding.csv')
args = parser.parse_args()
params = get_param_dict(args.setting_dir)
random_state = int(params['rand_seed'])
torch.manual_seed(random_state)
if params['gpu'] == '-1':
    params['device'] = torch.device('cpu')
else:
    params['device'] = torch.device('cuda:'+str(params['gpu']) if torch.cuda.is_available() else 'cpu')

# define run_attention_analysis
def run_embedding(dataset, embedding_pt_dir = '/amax/data/ruijin/embedding/ruijin_63/splita_person', params=dict(), keep_idx=None):
    '''
    dataset: patient dataset
    '''
    device = params['device']
    ckpt = params['ckpt']
    task = params['task']
    perslide = params['perslide']

    # complement the parameters in params dictionary
    data = dataset.__getitem__(0)
    patho = data['pathology']
    gene = data['gene']
    clinical = data['clinical']
    params['patho_embedding_dim']  = int(patho.shape[-1])
    params['gene_set_num'] = int(len(gene))
    params['gene_set_dim'] = [gene_tensor.shape[0] for gene_tensor in gene]

    # get specific MCAT model according to the task
    assert task in ['survival', 'stage']
    if task == 'survival':
        model = MTMCAT_survival(params=params)
    elif task == 'stage':
        model = MTMCAT(params=params)
    model = load_model(model, checkpoint_name=ckpt)
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if keep_idx != None:
        assert keep_idx < len(dataset)
    else:
        keep_idx = len(dataset)
    with torch.no_grad():
        model.eval()
        for idx, data in enumerate(dataloader):
            if idx < keep_idx:
                embedding = model.get_embedding(data)
                case_id = data['case_id'][0]
                if perslide == 'True':
                    slide_name = data['slide_name'][0]
                    pt_file_name = os.path.join(embedding_pt_dir, slide_name+'.pt') 
                else:
                    pt_file_name = os.path.join(embedding_pt_dir, case_id+'.pt')
                torch.save(embedding, pt_file_name)
                print('idx', idx)
                pass
    pass


# execute main function
if __name__ == '__main__':
    print('Start')
    start = time.time()
    patient_dataset = PatientDataset(params=params)
    if str(params['split_file']) != 'nan':
        if os.path.exists(params['split_file']):
            split_file = pd.read_csv(params['split_file'])
            print('split_file', params['split_file'])
            patient_list = list(split_file['train'].dropna()) + list(split_file['validation'].dropna()) + list(split_file['test'].dropna())
            # patient_list = list(split_file['train'].dropna())
            # patient_list = list(split_file['validation'].dropna()) + list(split_file['test'].dropna()) 
            patient_dataset = PatientDataset(params=params, patient_keep_list=patient_list)
    assert params['perslide'] in ['True', 'False']
    if params['perslide'] == 'True':
        tensor_dataset = PerslideDataset(patient_dataset, params=params, isTrain=False)
    else:
        tensor_dataset = MtmcatDataset(patient_dataset, params=params, isTrain=True)
    model_dir = './saved_model/' + params['ckpt'] + '.pth.tar'
    
    run_embedding(tensor_dataset, params=params, embedding_pt_dir='/amax/data/ruijin/embedding/ruijin_63/splita_person')  

    end = time.time()
    print(f'Program end, total running time {end - start} s')
    
