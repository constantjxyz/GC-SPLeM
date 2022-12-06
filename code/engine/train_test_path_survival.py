'''
    train_test.py: the training, validation and testing process
'''
# import dependent modules
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
from model.gene import GeneNet
from model.model_path_clam_survival import CLAM_path as MTMCAT
from engine.loss import cross_entropy_sur_loss, nll_sur_loss
from engine.utils import make_weights_for_balanced_classes_split, EarlyStopping
# training process
def train_path_survival(train_dataset, val_dataset, test_dataset, params=dict(), val=True):
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
    model = MTMCAT(params=params)
    model = model.to(device)
    # print('model', model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3,)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
    if params['use_wandb'] == 'True':
        wandb.watch(model, log='all') 

    if params['early_stopping'] == 'True':
        early_stopping = EarlyStopping()
    else:
        early_stopping = None

    # get the dataloader
    weights = make_weights_for_balanced_classes_split(train_dataset)
    print('weights of different class samples', np.unique(np.array(weights)))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, 
        sampler = WeightedRandomSampler(weights, len(weights)))
    
    # run epochs
    best_val_loss, best_epoch = 100.0, 0

    #random baseline wandb
    print('now starts the validation process of epoch 0')
    val_loss, val_acc, val_c_index= test_path_survival(val_dataset, model, params=params, mode='val')
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

    for epoch in range(1, epochs+1):
        model.train()
        train_loss_all = []
        all_risk_scores =[]
        all_censorships = []
        all_event_times = []
        all_Y_hat_labels = []
        all_disc_labels = []
        case_risk_dict = {}
        event_time_dict = {}
        censorship_dict = {}
        for i, data in enumerate(train_dataloader):     
            model.train()
            if params['use_inst'] == 'True':
                hazards, S, Y_hat, c, Y, event_time, total_inst_loss = model(data)
            else:
                hazards, S, Y_hat, c, Y, event_time, = model(data)
            
            # hazards: tensor shape: (1, n_class)
            # Y_hat: tensor shape: (1,1)
            # S: tensor shape:(1, n_class)
            # c: tensor shape:(1)
            if params['loss_type'] == 'nll':
                loss = nll_sur_loss(hazards, S, Y, c, alpha=alpha, eps=eps)
            elif params['loss_type'] == 'ce':
                loss= cross_entropy_sur_loss(hazards, S, Y, c, alpha=alpha, eps=eps)
            else:
                raise NotImplementedError(params['loss_type'])
            # loss: tensor shape(1), on device
            if params['use_inst'] == 'True':
                loss += total_inst_loss
            loss_value = loss.item()
            risk = torch.sum(hazards).detach().cpu().numpy()
            all_risk_scores.append(risk.item())
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            all_disc_labels.append(Y.reshape(1).item())
            all_Y_hat_labels.append(Y_hat.reshape(1).item())
            train_loss_all.append(loss_value)


            # backward pass
            loss = loss / gc
            loss.backward()
            if (i + 1) % gc == 0: 
                optimizer.step()
                optimizer.zero_grad()
                # print(f'batch {i+1}, {len(train_dataloader)} all')
            
            # case id record
            case_id = data['case_id'][0]
            if case_id not in case_risk_dict:
                case_risk_dict[case_id] = [risk.item()]
                event_time_dict[case_id] = [event_time.item()]
                censorship_dict[case_id] = [c.item()]
            else:
                case_risk_dict[case_id] += [risk]
  
        
        all_risk_scores = np.array(all_risk_scores)
        all_censorships = np.array(all_censorships)
        all_event_times = np.array(all_event_times)
        all_Y_hat_labels = np.array(all_Y_hat_labels)
        all_disc_labels = np.array(all_disc_labels)

        # evaluation metrics
        train_loss_average = np.mean(np.array(train_loss_all))
        train_acc = accuracy_score(all_disc_labels, all_Y_hat_labels)
        # print('train classification report')
        # print(classification_report(all_disc_labels, all_Y_hat_labels, zero_division=0))
        # train_c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
        train_c_index = np.float32(concordance_index_censored((1-all_censorships).astype(bool), all_event_times, np.float32(all_risk_scores), tied_tol=1e-08)[0])

        print(f'epoch: {str(epoch)}/{str(epochs)}, training loss:{train_loss_average}, c index:{train_c_index}, train accuracy:{train_acc}')

        # perslide training result
        if params['perslide'] == 'True':
            if True:
                censorship_all_case = []
                mean_predict_risk_all_case = []
                max_predict_risk_all_case = []
                event_time_all_case = []
                slides_c_index = train_c_index
                for key in case_risk_dict.keys():
                    predict = case_risk_dict[key]
                    censorship = censorship_dict[key]
                    event_time = event_time_dict[key]
                    censorship_all_case.append(censorship)
                    event_time_all_case.append(event_time)
                    mean_predict_risk_all_case.append(np.array(predict).mean())
                    max_predict_risk_all_case.append(np.array(predict).max())
            
            censorship_all_case = np.array(censorship_all_case).reshape(-1)
            mean_predict_risk_all_case = np.array(mean_predict_risk_all_case).reshape(-1)
            max_predict_risk_all_case = np.array(max_predict_risk_all_case).reshape(-1)
            event_time_all_case = np.array(event_time_all_case).reshape(-1)

            mean_case_c_index = np.float32(concordance_index_censored((1-censorship_all_case).astype(bool), event_time_all_case, np.float32(mean_predict_risk_all_case), tied_tol=1e-08)[0]) 
            max_case_c_index = np.float32(concordance_index_censored((1-censorship_all_case).astype(bool), event_time_all_case, np.float32(max_predict_risk_all_case), tied_tol=1e-08)[0] )
            train_c_index = [slides_c_index, mean_case_c_index, max_case_c_index]

            print(f'Train dataset perslide test, mean case test c index {train_c_index[1]}, max case test c index {train_c_index[2]}') 

        # val_c_index: float number or list[slides_c_index, mean_case_c_inex, max_case_c_index]
        val_loss, val_acc, val_c_index= test_path_survival(val_dataset, model, params=params, mode='val')
        if params['perslide'] == 'False':
            print(f'epoch: {str(epoch)}/{str(epochs)}, val loss:{val_loss}, c index:{val_c_index}, val accuracy:{val_acc}')
        else:
            print(f'epoch: {str(epoch)}/{str(epochs)}, val loss:{val_loss}, slides c index:{val_c_index[0]}, val accuracy:{val_acc}')
            print(f'Validation dataset perslide test, mean case test c index {val_c_index[1]}, max case test c index {val_c_index[2]}') 


        if params['use_wandb'] == 'True':
            if params['perslide'] == 'False':
                wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss_average,
                "Train C_index": train_c_index,
                "Train Accuracy": train_acc, 
                "Validation Loss": val_loss,
                "Validation C_index": val_c_index,
                "Validation Accuracy": val_acc,
                })
            else:
                wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss_average,
                "Train slides C_index": train_c_index[0],
                "Train Accuracy": train_acc,
                "Train mean case C_index": train_c_index[1], 
                "Train max case C_index": train_c_index[1],
                "Validation Loss": val_loss,
                "Validation slides C_index": val_c_index[0],
                "Validation Accuracy": val_acc,
                "Validation mean case C_index": val_c_index[1], 
                "Validation max case C_index": val_c_index[2],
                })

        # whether to save the new model
        if early_stopping:
            early_stopping(epoch, val_loss, model, ckpt_name=ckpt)

        elif val_loss < (best_val_loss):
            save_model(model, checkpoint_name=ckpt)
            best_val_loss = val_loss
            best_epoch = epoch
        
        elif  ((val_loss - best_val_loss) / best_val_loss) < 0.05 and (epoch - best_epoch) > 5 and epoch > 10:
            # save models close the best val loss because of the fluctuation
            save_model(model, checkpoint_name=ckpt+'_epoch'+str(epoch)) 
            
        # print out an empty line to separate different epochs
        print('')
    print(f'best validation loss:{str(best_val_loss)}, in epoch: {str(best_epoch)}/{str(epochs)}')
    return model, best_val_loss


# testing process
def test_path_survival(test_dataset, model, params=dict(), mode='test'):
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
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # testing process
    with torch.no_grad():
        # total, correct = 0, 0
        model.eval()
        test_loss_all = []
        all_risk_scores =[]
        all_censorships = []
        all_event_times = []
        all_Y_hat_labels = []
        all_disc_labels = []
        case_risk_dict = {}
        event_time_dict = {}
        censorship_dict = {}

        for i, data in enumerate(test_dataloader):
            if params['use_inst'] == 'True':
                hazards, S, Y_hat, c, Y, event_time, total_inst_loss = model(data)
            else:
                hazards, S, Y_hat, c, Y, event_time = model(data)
            # hazards: tensor shape: (1, n_class)
            # Y_hat: tensor shape: (1,1)
            # S: tensor shape:(1, n_class)
            # c: tensor shape:(1)
            if params['loss_type'] == 'nll':
                loss = nll_sur_loss(hazards, S, Y, c, alpha=alpha, eps=eps)
            elif params['loss_type'] == 'ce':
                loss= cross_entropy_sur_loss(hazards, S, Y, c, alpha=alpha, eps=eps)
            else:
                raise NotImplementedError(params['loss_type'])
            
            # loss: tensor shape(1), on device
            if params['use_inst'] == 'True':
                loss += total_inst_loss
            loss_value = loss.item()
            risk = torch.sum(hazards).detach().cpu().numpy()
            all_risk_scores.append(risk.item())
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            all_disc_labels.append(Y.reshape(1).item())
            all_Y_hat_labels.append(Y_hat.reshape(1).item())
            test_loss_all.append(loss_value) 

            case_id = data['case_id'][0]
            if case_id not in case_risk_dict:
                case_risk_dict[case_id] = [risk.item()]
                event_time_dict[case_id] = [event_time.item()]
                censorship_dict[case_id] = [c.item()]
            else:
                case_risk_dict[case_id] += [risk]
        
        
        all_risk_scores = np.array(all_risk_scores)
        all_censorships = np.array(all_censorships)
        all_event_times = np.array(all_event_times)
        all_Y_hat_labels = np.array(all_Y_hat_labels)
        all_disc_labels = np.array(all_disc_labels)

        

        # evaluation metrics
        test_loss_average = np.mean(np.array(test_loss_all))
        test_acc = accuracy_score(all_disc_labels, all_Y_hat_labels)
        # print('test classification report')
        # print(classification_report(all_disc_labels, all_Y_hat_labels, zero_division=0))
        # test_c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
        test_c_index = np.float32(concordance_index_censored((1-all_censorships).astype(bool), all_event_times, np.float32(all_risk_scores), tied_tol=1e-08)[0])

        if mode == 'test':
            print(f'Test dataset, test loss {test_loss_average}, test c index {test_c_index}, test accuracy {test_acc}')

        # per slide test
        if params['perslide'] == 'True':
            if True:
                censorship_all_case = []
                mean_predict_risk_all_case = []
                max_predict_risk_all_case = []
                event_time_all_case = []
                slides_c_index = test_c_index
                for key in case_risk_dict.keys():
                    predict = case_risk_dict[key]
                    censorship = censorship_dict[key]
                    event_time = event_time_dict[key]
                    censorship_all_case.append(censorship)
                    event_time_all_case.append(event_time)
                    mean_predict_risk_all_case.append(np.array(predict).mean())
                    max_predict_risk_all_case.append(np.array(predict).max())
            
            censorship_all_case = np.array(censorship_all_case).reshape(-1)
            mean_predict_risk_all_case = np.array(mean_predict_risk_all_case).reshape(-1)
            max_predict_risk_all_case = np.array(max_predict_risk_all_case).reshape(-1)
            event_time_all_case = np.array(event_time_all_case).reshape(-1)

            mean_case_c_index = np.float32(concordance_index_censored((1-censorship_all_case).astype(bool), event_time_all_case, np.float32(mean_predict_risk_all_case), tied_tol=1e-08)[0] )
            max_case_c_index = np.float32(concordance_index_censored((1-censorship_all_case).astype(bool), event_time_all_case, np.float32(max_predict_risk_all_case), tied_tol=1e-08)[0]) 
            test_c_index = [slides_c_index, mean_case_c_index, max_case_c_index]
            
            if mode == 'test':
                print(f'Test dataset perslide test, mean case test c index {test_c_index[1]}, max case test c index {test_c_index[2]}')
     


    # return loss_average, correct/total
    # test_c_index: float number or list[slides_c_index, mean_case_c_inex, max_case_c_index]
    return test_loss_average, test_acc, test_c_index
