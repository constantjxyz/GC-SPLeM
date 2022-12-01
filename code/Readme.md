Here we show the pipeline of our work.

1. Use the WSI feature extractor. Follow the work of CLAM 'https://github.com/mahmoodlab/CLAM'. Save the extracted feature matrix of WSIs.

2. Train the modal-fusing network:

	(a) Deal with the original data. Processed data saved in './dataset' folder.

	(b) Set a setting file in './setting' folder.

	(c) Run 'main.py' in python with a specific setting file.

	(d) Functions of some important python files. './dataset/create_dataset.py': loading the datasets; './engine/run_train_test.py': splitting the dataset; './engine/train_test_survival': training the modal-fusing network; './model/model_mtMCAT_survival': modal-fusing network model; 
	
	(e) Final model saved in './saved_model' folder.

3. Obtain the final embedding of modal-fusing network using 'get_embedding.py'.

4. Create the graph of patients using './dataset/xx/others/create_feature_adj.ipynb'. Save the created '.h5' file.

5. Train the GNN-based predictor:
	
	(a) Set a setting file in the './setting' folder.

	(b) Run 'main_gcn.py' with python.

	(c) Functions of some important python files. './dataset/create_dataset.py': loading the datasets; './engine/run_train_test.py': splitting the dataset; './engine/train_test_survival': training the modal-fusing network; './model/model_gnn_survival': GNN-based predictor model;

	(d) Final model saved in the './saved_model' folder.

6. Compare with other methods:

	(a) MCAT: ignore the GNN-based predictor.

	(b) WSI-only: use settings marked as 'path'.

	(c) GC-SPLeM without GNN: delete the two graph convolution layers in './model/model_gnn_survival'.

7. Train the incomplete dataset: use settings marked with 'pool'.

8. Important tools and their versions: torch 1.7.1+cu101; scikit-learn 1.0.1; openslide-python 1.1.2; scikit-survival 0.16.0; GeForce RTX 2080Ti GPU device with CUDA 10.1;