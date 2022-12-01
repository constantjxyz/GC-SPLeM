
# import dependencies from module
import argparse
import time
import torch
import wandb
import os

# import dependencies from folder
from setting.get_parameter import get_param_dict
from dataset.create_gnn_dataset import PatientDataset
from engine.run_train_test_gnn import run_cross_validation
from sklearn.model_selection import train_test_split

# set parameters from csv
parser = argparse.ArgumentParser()
parser.add_argument('--setting_dir', type=str, default='./setting/gnn/ruijin_63/gnn_pool.csv')
args = parser.parse_args()
params = get_param_dict(args.setting_dir)
random_state = int(params['rand_seed'])
torch.manual_seed(random_state)
if params['gpu'] == '-1':
    params['device'] = torch.device('cpu')
else:
    params['device'] = torch.device('cuda:'+str(params['gpu']) if torch.cuda.is_available() else 'cpu')

# write down the experiments by using wandb
if params['use_wandb'] == 'True':
    wandb.init(project='ruijin_63_gnn_random', name=params['ckpt'], entity="sjtu_mtmcat")

# define main function
def main():
    # get the dataset
    dataset = PatientDataset(params=params)
    # split the dataset
    assert params['stratify'] in ['True', 'False']
    if params['stratify'] == 'True':
        train_index, test_index = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.return_key_weight_label(), random_state=random_state)
    else:
        train_index, test_index = train_test_split(range(len(dataset)), test_size=0.2, random_state=random_state)
    train_dataset = PatientDataset(params=params, patient_keep=train_index)
    test_dataset = PatientDataset(params=params, patient_keep=test_index)
    # put the dataset to the engine
    run_cross_validation(train_dataset, test_dataset, params=params)
    

# execute main function
if __name__ == '__main__':
    print('Start')
    start = time.time()
    print('setting file', args.setting_dir)
    main()
    end = time.time()
    print(f'Program end, total running time {end - start} s')