Set all the hyper-parameters here

###################### files ####################################

'get_paramter.py': read all the settings in each setting file

setting files: format as '.csv'

'modal_fusing.csv': setting of training the modal-fusing network

'gnn.csv': setting of training the GNN-based predictor

'path_clam_surv': setting of training the single-modality survival prediction model

###################### hyper-parameters for training modal-fusing network ###########################

dataset_dir: the direction of dataset, a folder containing 'gene.csv', 'pathology.csv', 'patient.csv', and 'clinical.csv'

wsi_feature_dir: the direction of saved WSI-feature matrices, on your own device

split_file: the '.csv' file indicating split of the train, validation, and test set

save_split_dir: the direction to save split

logit_dir: not used

ckpt: the folder to save your trained model. For example, if you write 'luad/splita' here, the trained model will be saved in './saved_model/luad/splita'

task: usually set as 'survival'

cross_validation_folds: set as 1

epochs, learning_rate: training epochs and learning rate

batch_size, gradient_accumulation: batch_size need to set as 1 because different samples have different patch number, gradient_accumulation is set to achieve batch learning 

loss_type, loss_alpha, loss_eps: indicate loss funcion

loss_mode, early_stopping, stratify, perslide, shuffle_mode, shuffle_scope, shuffle_type, multi-instance_mode: not used, just keep the original setting here is ok

gpu: which gpu to use, write a number: '-1' denotes using the cpu, '0' denotes the'gpu:0'

use_wandb: whether use wandb for tracking training results

other parameters are about the model's structure, you can just keep the same


###################### hyper-parameters for training GNN-based predictor###########################

h5_dataset_dir: the direction of saved embedding matrix of graph; the graph is create by running './dataset/xx/others/create_feature_adj.ipynb'

epochs, learning_rate batch_size, gradient_accumulation: need to adjust to GNN training

loss_alpha, loss_eps, loss_type: keep the same loss function as training the modal-fusing network

###################### hyper-parameters for training WSI-only model###########################

task: set as 'path_survival'

use_inst: set as 'True'

###################### hyper-parameters for training incomplete dataset###########################

shuffle_type: set as 'gene'

shuffle_mode: set as 'random_50%'

shuffle_scope: set as 'train'; generate random values for some samples in the training set

