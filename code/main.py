'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-02-11 15:53:47
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-09-06 19:00:38
FilePath: /mtmcat/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# import dependencies from module
import argparse
import time
import torch
import wandb
import os

# import dependencies from folder
from setting.get_parameter import get_param_dict
from dataset.create_dataset import PatientDataset
from engine.run_train_test import run_cross_validation
from sklearn.model_selection import train_test_split

# set parameters from csv
parser = argparse.ArgumentParser()
parser.add_argument('--setting_dir', type=str, default='./setting/ebio/ln/ln_ori_noco.csv')
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
    wandb.init(project='ebio_ln_noco', name=params['ckpt'], entity="sjtu_mtmcat")


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

