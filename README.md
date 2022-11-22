# Convex Optimization based Autoencoder Neural Networks
# based on the paper
# V. Gupta, B. Bartan, T. Ergen, M. Pilanci
# Convex Neural Autoregressive Models: Towards Tractable, Expressive, and Theoretically-Backed Models for Sequential Forecasting and Generation
# IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021 
# (Outstanding Student Paper Award)

## Dependencies
Python 3.7.4

PyTorch 1.4.0

NumPy 1.18.1

CVXPY 1.0.31

scikit-learn 0.22.1

Matplotlib 3.1.1

## Code
The binary vector data is in the 'all_data' folder (obtained by downloading the code [here](http://info.usherbrooke.ca/hlarochelle/code/nade.tar.gz) and running download_datasets.py using Python 2). Data is pre-processed using the functions in 'utils/preprocess.py', specifically the `preprocess_data` function. Dataloaders for the PyTorch models are created via the `get_dataloader` method in 'utils/helpers.py'.

Code for generating the random vectors $u_j$ and corresponding diagonal matrices $D_j$ for the convex models is in 'utils/helpers.py'. 

Code for training and evaluating the CVXPY models is in 'utils/cvxpy_model.py', which contains a class for evaluation and CVXPY solver. 

Code for the PyTorch models is in 'utils/pytorch_model.py', which contains a PyTorch Layer class for the convex model and a generic SGD solver. Helpers for calculating the loss and accuracy for the PyTorch models are contained in 'utils/helpers.py', while those for calculating the negative log-likelihood are in 'utils/loss.py'. Helpers for printing information about the PyTorch models are also in 'utils/helpers.py'. 


## Running the Models

### Script
In order to run the 5 models on a dataset 10 times, run `python run_models.py n`, where `n` is the index of the dataset to run the model on in the following array: ['adult', 'connect4', 'dna', 'mushrooms', 'nips', 'ocr', 'rcv1', 'web']. To change the number of times you run these 5 models, change the variable `num_runs` in 'run_models.py'. To change the hyperparameters of the models, you must manually change them in 'utils/run_model_helpers.py'. The output of each run (a few summary statistics) will be in the 'outputs' folder under the corresponding dataset. For instance, the statistics for the tables in the supplement can be generated by running the above command for `n` = 0, ..., 7.


### Jupyter Notebooks
There are Jupyter notebooks for every dataset in the base directory. They can be used for hyperparameter tuning, or for visualizing the results, such as showing images for binarized MNIST or plotting the training curves and time breakdowns. The file 'utils/visualization.py' contains these methods.
