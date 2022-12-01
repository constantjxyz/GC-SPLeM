Setting all the hyper-parameters here

###################### files ####################################
'get_paramter.py': read all the settings in each setting file

setting files: format as '.csv'

'modal_fusing.csv': setting of training the modal-fusing network

'gnn.csv': setting of training the GNN-based predictor

'path_clam_surv': setting of training the single-modality survival prediction model

###################### hyper-parameters ###########################
dataset_dir: the direction of dataset, a folder containing 'gene.csv', 'pathology.csv', 'patient.csv', and 'clinical.csv'

wsi_feature_dir: the direction of saved WSI-feature matrices, on your own device

split_file: the '.csv' file indicating split of the train, validation, and test set

save_split_dir: the direction to save split

logit_dir: not used

ckpt: the folder to save your trained model. For example, if you write 'luad/splita' here, the trained model will be saved in './saved_model/luad/splita'

task: usually set as 'survival', sometimes set as 'path_survival' for using WSI-only survival prediction task

cross_validation_folds



