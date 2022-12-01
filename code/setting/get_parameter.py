'''idea:
        get the setting csv, return a dictionary to the main function
        the dictionary can be delivered to other python files(like engine.py), and can be altered
'''

# import dependent module
import numpy as np
import pandas as pd
import os

# define the function 'get_param_dict'
'''input: the param directory(csv file)
   output: a dictionary, dict.keys() for parameter name, dict.values() for parameter specific settings
'''
def get_param_dict(param_dir):
    assert param_dir.split('.')[-1] == 'csv'
    file = pd.read_csv(param_dir)
    d = dict()
    for i in range(len(file)):
        name = file.loc[i, 'name']
        setting = file.loc[i, 'setting']
        d[name] = setting

    # add auto/None ckpt

    if 'ckpt' not in d.keys() or d['ckpt'] == 'auto':
        d['ckpt'] = '%s_lr%s_cv%s_rand%s_%s_acc' % \
            (param_dir.split('/')[-1].split('.')[0], d['learning_rate'], d['cross_validation_folds'], d['rand_seed'], d['multi_instance_mode'])
        if d['loss_type'] == 'focal':
            d['ckpt'] += '_focal%s_%s' % (d['focal_alpha'], d['focal_gamma'])
        elif d['loss_type'] == 'ce':
            d['ckpt'] += '_ce'
        elif d['loss_type'] == 'nll':
            d['ckpt'] += '_nll'
        if d['shuffle_type'] != 'False':
            d['ckpt'] += '_shuffle_%s_%s' % (d['shuffle_type'], d['shuffle_mode'])
            if d['shuffle_scope'] == 'train':
                d['ckpt'] += '_train_scope'
    
        print('checkpoint name change: %s' % d['ckpt'])

    return d

