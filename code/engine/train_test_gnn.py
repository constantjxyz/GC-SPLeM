# import dependent modules
from tkinter import Y
import torch
import numpy as np
import os
import wandb
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import accuracy_score, classification_report
# from lifelines.utils import concordance_index

# import dependent functions from utils.py
from engine.utils import save_model, load_model
from engine.loss import cross_entropy_sur_loss, nll_sur_loss, neg_log_partial_likelihood_with_reset_index
from engine.utils import make_weights_for_balanced_classes_split
from model.model_gnn_survival import GCN_survival

# training process
def train_survival(train_dataset, val_dataset, test_dataset, params=dict(), val=True):
    '''
    the training epochs of train_datset and val_dataset
    val: can be True of False, if False, no need to carry on the validation process
    '''
    device = params['device']
    epochs = int(params['epochs'])
    lr = float(params['learning_rate'])
    batch_size = int(params['batch_size'])
    ckpt = params['ckpt']
    alpha = float(params['loss_alpha'])
    eps = float(params['loss_eps'])
    gc = int(params['gradient_accumulation'])
    model = GCN_survival(params=params)
    model = model.to(device)
    # print('model', model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3,)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
    if params['use_wandb'] == 'True':
        wandb.watch(model, log='all') 
    
    # get the dataloader
    weights = make_weights_for_balanced_classes_split(train_dataset)
    # print('weights of different class samples', np.unique(np.array(weights)))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler = WeightedRandomSampler(weights, len(weights)))
    
    # run epochs
    best_val_loss, best_epoch = 100.0, 0

    #random baseline wandb
    print('now starts the validation process of epoch 0')
    val_loss, val_acc, val_c_index= test_survival(val_dataset, model, params=params, mode='val')
    if params['use_wandb'] == 'True':
        if params['perslide'] == 'False':
            wandb.log({
                "Epoch": 0,
                "Validation Loss": val_loss,
                "Validation C_index": val_c_index,
                "Validation Accuracy": val_acc,
                })
        else:
            wandb.log({
                "Epoch": 0,
                "Validation Loss": val_loss,
                "Validation slides C_index": val_c_index[0],
                "Validation Accuracy": val_acc,
                "Validation mean case C_index": val_c_index[1], 
                "Validation max case C_index": val_c_index[2],
            })
    
    test_loss, test_acc, test_c_index= test_survival(test_dataset, model, params=params, mode='val')
    if params['use_wandb'] == 'True':
        if params['perslide'] == 'False':
            wandb.log({
                "Epoch": 0,
                "Test Loss": test_loss,
                "Test C_index": test_c_index,
                "Test Accuracy": test_acc,
                })
        else:
            wandb.log({
                "Epoch": 0,
                "Test Loss": test_loss,
                "Test slides C_index": test_c_index[0],
                "Test Accuracy": test_acc,
                "Test mean case C_index": test_c_index[1], 
                "Test max case C_index": test_c_index[2],
            })
    
    for epoch in range(1, epochs+1):
        model.train()
        all_case_ids = []
        train_loss_all = []
        all_risk_scores =[]
        all_censorships = []
        all_event_times = []
        all_Y_hat_labels = []
        all_disc_labels = []
        
        features, adj = train_dataset.return_features_adj()
        features, adj  = torch.from_numpy(features), torch.from_numpy(adj)
        for i, data_dict in enumerate(train_dataloader):     
            cases_all_hazards, cases_all_S, cases_all_Y_hat, cases_all_logits= model(features, adj)
            # hazards: tensor shape: (nodes_num, n_class)
            # Y_hat: tensor shape: (nodes_num,1)
            # S: tensor shape:(nodes_num, n_class)
            # all_logits shape: (nodes_num, n_class)
            # get all the index of batch cases
            batch_cases_idx = data_dict['case_id_idx'] 
            batch_cases_id = data_dict['case_id']   # list
            batch_hazards = cases_all_hazards[batch_cases_idx]
            batch_Y_hat = cases_all_Y_hat[batch_cases_idx]
            batch_S = cases_all_S[batch_cases_idx]
            batch_Y = data_dict['clinical'][2].to(device)
            batch_c = data_dict['clinical'][0].to(device)
            batch_event_days = data_dict['clinical'][1].to(device)
            if params['loss_type'] == 'nll':
                loss = nll_sur_loss(batch_hazards, batch_S, batch_Y, batch_c, alpha=alpha, eps=eps)
            elif params['loss_type'] == 'ce':
                loss= cross_entropy_sur_loss(batch_hazards, batch_S, batch_Y, batch_c, alpha=alpha, eps=eps)
            elif params['loss_type'] == 'neg':
                loss = neg_log_partial_likelihood_with_reset_index(batch_event_days, cases_all_logits)
                # loss = neg_log_partial_likelihood_with_reset_index(batch_event_days, cases_all_hazards) 
            else:
                raise NotImplementedError(params['loss_type'])
            # loss: tensor shape(1), on device
            loss_value = loss.item()
            # risk = torch.sum(batch_hazards, dim=1).detach().cpu().numpy()
            # case_id = data_dict['case_id'][0]
            all_risk_scores += list(torch.sum(batch_hazards, dim=1).detach().cpu().numpy())
            all_censorships += list(batch_c.cpu().numpy())
            all_event_times += list(batch_event_days.cpu().numpy())
            all_disc_labels += list(batch_Y.cpu().numpy())
            all_Y_hat_labels += list(batch_Y_hat.cpu().numpy())
            all_case_ids += batch_cases_id
            train_loss_all.append(loss_value)


            # backward pass
            loss = loss / gc
            loss.backward()
            if (i + 1) % gc == 0: 
                optimizer.step()
                optimizer.zero_grad()
                # print(f'batch {i+1}, {len(train_dataloader)} all')
        
        all_risk_scores = np.array(all_risk_scores)
        all_censorships = np.array(all_censorships)
        all_event_times = np.array(all_event_times)
        all_Y_hat_labels = np.array(all_Y_hat_labels)
        all_disc_labels = np.array(all_disc_labels)
        all_case_ids = np.array(all_case_ids)

        # evaluation metrics
        train_loss_average = np.mean(np.array(train_loss_all))
        train_acc = accuracy_score(all_disc_labels, all_Y_hat_labels)
        # print('train classification report')
        # print(classification_report(all_disc_labels, all_Y_hat_labels, zero_division=0))
        # train_c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
        train_c_index = np.float32(concordance_index_censored((1-all_censorships).astype(bool), all_event_times, np.float32(all_risk_scores), tied_tol=1e-08)[0])

        print(f'epoch: {str(epoch)}/{str(epochs)}, training loss:{train_loss_average}, c index:{train_c_index}, train accuracy:{train_acc}')

        val_loss, val_acc, val_c_index= test_survival(val_dataset, model, params=params, mode='val')
        print(f'epoch: {str(epoch)}/{str(epochs)}, val loss:{val_loss}, c index:{val_c_index}, val accuracy:{val_acc}')

        '''---------------------------------------------------'''
        test_loss, test_acc, test_c_index = test_survival(test_dataset, model, params=params, mode='val')
        # print(f'epoch: {str(epoch)}/{str(epochs)}, test loss:{test_loss}')
        # print('')
        '''---------------------------------------------------'''
        if params['use_wandb'] == 'True':
             wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss_average,
                "Train C_index": train_c_index,
                "Train Accuracy": train_acc, 
                "Validation Loss": val_loss,
                "Validation C_index": val_c_index,
                "Validation Accuracy": val_acc,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Test C_index": test_c_index,
                })

        # whether to save the new model
        if val_loss < (best_val_loss):
            save_model(model, checkpoint_name=ckpt)
            best_val_loss = val_loss
            best_epoch = epoch
        
        # elif  ((val_loss - best_val_loss) / best_val_loss) < 0.05 and (epoch - best_epoch) > 5 and epoch > 10:
        #     # save models close the best val loss because of the fluctuation
        #     save_model(model, checkpoint_name=ckpt+'_epoch'+str(epoch))

        # print out an empty line to separate different epochs
        print('')
    print(f'best validation loss:{str(best_val_loss)}, in epoch: {str(best_epoch)}/{str(epochs)}')
    return model, best_val_loss

# testing process
def test_survival(test_dataset, model, params=dict(), mode='test'):
    '''
    mode: can be 'test' or 'val', indicating test or validation process
    '''
    device = params['device']
    ckpt = params['ckpt']
    alpha = float(params['loss_alpha'])
    eps = float(params['loss_eps'])

    # need to use the saved best model when testing, need to use the current model when validating
    if mode == 'test':
        model = load_model(model, checkpoint_name=ckpt)
        model = model.to(device)
    
    # get the test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # testing process
    with torch.no_grad():
        # total, correct = 0, 0
        model.eval()
        all_case_ids = []
        test_loss_all = []
        all_risk_scores =[]
        all_censorships = []
        all_event_times = []
        all_Y_hat_labels = []
        all_disc_labels = []
        
        features, adj = test_dataset.return_features_adj()
        features, adj  = torch.from_numpy(features), torch.from_numpy(adj)
        for i, data_dict in enumerate(test_dataloader):     
            cases_all_hazards, cases_all_S, cases_all_Y_hat, cases_all_logits= model(features, adj)
            # hazards: tensor shape: (nodes_num, n_class)
            # Y_hat: tensor shape: (nodes_num,1)
            # S: tensor shape:(nodes_num, n_class)
            # all_logits shape: (nodes_num, n_class)
            # get all the index of batch cases       
            batch_cases_idx = data_dict['case_id_idx'] 
            batch_cases_id = data_dict['case_id']   # list
            batch_hazards = cases_all_hazards[batch_cases_idx]
            batch_Y_hat = cases_all_Y_hat[batch_cases_idx]
            batch_S = cases_all_S[batch_cases_idx]
            batch_Y = data_dict['clinical'][2].to(device)
            batch_c = data_dict['clinical'][0].to(device)
            batch_event_days = data_dict['clinical'][1].to(device) 

            if params['loss_type'] == 'nll':
                loss = nll_sur_loss(batch_hazards, batch_S, batch_Y, batch_c, alpha=alpha, eps=eps)
            elif params['loss_type'] == 'ce':
                loss= cross_entropy_sur_loss(batch_hazards, batch_S, batch_Y, batch_c, alpha=alpha, eps=eps)
            elif params['loss_type'] == 'neg':
                loss = neg_log_partial_likelihood_with_reset_index(batch_event_days, cases_all_logits)
                # loss = neg_log_partial_likelihood_with_reset_index(batch_event_days, cases_all_hazards) 
            else:
                raise NotImplementedError(params['loss_type'])
            # loss: tensor shape(1), on device
            loss_value = loss.item()
            # risk = torch.sum(batch_hazards, dim=1).detach().cpu().numpy()
            # case_id = data_dict['case_id'][0]
            all_risk_scores += list(torch.sum(batch_hazards, dim=1).detach().cpu().numpy())
            all_censorships += list(batch_c.cpu().numpy())
            all_event_times += list(batch_event_days.cpu().numpy())
            all_disc_labels += list(batch_Y.cpu().numpy())
            all_Y_hat_labels += list(batch_Y_hat.cpu().numpy())
            all_case_ids += batch_cases_id
            test_loss_all.append(loss_value)

        all_risk_scores = np.array(all_risk_scores)
        all_censorships = np.array(all_censorships)
        all_event_times = np.array(all_event_times)
        all_Y_hat_labels = np.array(all_Y_hat_labels)
        all_disc_labels = np.array(all_disc_labels)
        all_case_ids = np.array(all_case_ids)

        

        # evaluation metrics
        test_loss_average = np.mean(np.array(test_loss_all))
        test_acc = accuracy_score(all_disc_labels, all_Y_hat_labels)
        # print('test classification report')
        # print(classification_report(all_disc_labels, all_Y_hat_labels, zero_division=0))
        # test_c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
        test_c_index = np.float32(concordance_index_censored((1-all_censorships).astype(bool), all_event_times, np.float32(all_risk_scores), tied_tol=1e-08)[0])

        if mode == 'test':
            print(f'Test dataset, test loss {test_loss_average}, test c index {test_c_index}, test accuracy {test_acc}')
    
    return test_loss_average, test_acc, test_c_index
