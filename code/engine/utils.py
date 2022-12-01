'''
some functions used in engine folder
'''

# import dependent module
import torch
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pandas as pd

# functions about save and load model
def save_model(model, checkpoint_name=''):
    filename = 'saved_model/' + checkpoint_name + '.pth.tar'
    torch.save({'state_dict': model.state_dict()}, filename)
    print('current best model saved')
    print(f'save best model in {filename}')
    print('')


def load_model(model, checkpoint_name=''):
    filename = 'saved_model/' + checkpoint_name + '.pth.tar'
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print(f'load model from {filename}')
    return model

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    # label_all = []
    # for idx in range(len(dataset)):   
    #     y = dataset.__getitem__(idx)['clinical'][0]                       
    #     label_all.append(y) 
    label_all = dataset.return_key_weight_label()
    counter = Counter(label_all)                                       
    weight_per_class = [ (N/counter[c]) for c in range(len(counter))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(label_all)):   
        y = label_all[idx]                     
        weight[idx] = weight_per_class[int(y)]                                  

    return torch.DoubleTensor(weight)

def check_split(train_dataset, val_dataset, test_dataset):
    '''
    check whether the three datasets do not overlap and they cover all the patients
    '''
    a, b, c = train_dataset.return_patient_index(), val_dataset.return_patient_index(), test_dataset.return_patient_index()
    set(a) & set(c)
    set(b) & set(c)
    set(a) & set(b)

def save_split(train_dataset, val_dataset, test_dataset, save_dir=''):
    train_list = train_dataset.return_patient_list()
    val_list = val_dataset.return_patient_list()
    test_list = test_dataset.return_patient_list()
    patient_export_file = pd.DataFrame(columns=['train', 'validation', 'test'])

    patient_export_file['train'] = train_list
    patient_export_file.loc[:len(val_list)-1, 'validation'] = val_list
    patient_export_file.loc[:len(test_list)-1, 'test'] = test_list
    patient_export_file.to_csv(save_dir, index=False)
    return 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
    
