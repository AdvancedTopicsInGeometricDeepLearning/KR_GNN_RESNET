# KR_GNN_RESNET
A repo that presents our experiments with using kernel
regression with graph neural networks while utilizing skip 
connections. This project houses the results for the paper.

## File structure

* `./configs` contains configurations for running the tests.
* `./dataset.py` contains functions for loading all the datasets that were used in the paper.
* `./fetch_results.py` contains a script for making a CSV file.
* `./gaussian_kernel.py` contains the logic for computing kernel regression.
* `./knn.py` contains ?
* `./model.py` contains the code for the model that is used in self supervised settings.
* `./requirements.txt` contains the requirements used for the project, though this list may be incomplete.
* `./run_exp.py` contains the code for running a specific model on a specific dataset, it is used by `./run_many_exp.py`.
* `./run_many_exp.py` runs all the model on all the datasets.

As for these 3, they appear to not use KR at all, and thus are the regular variants of the models.
* `./model_reg.py` pretty sure the reg here stand for regular, as in, KR was not used at all.
* `./run_exp_reg.py` contains the code for running a specific model on a specific dataset, it is used by `./run_many_exp_reg.py`.
* `./run_many_exp_reg.py` runs all the model on all the datasets.


## Getting dependencies

To get all dependencies, or just move on to the next step and install dependencies as they arise.
First, make a new miniconda environment that we called `GDL`:
```commandline
conda create --name GDL python=3.11
```
Then activate the environment:
```commandline
conda activate GDL
```
Then install some dependencies using conda:
```commandline
conda install -c pytorch -c nvidia -c pyg pytorch-sparse faiss-gpu lightning seaborn tensorboard
```
Then install the rest of the dependencies using pip:
```commandline
pip install torch-geometric wandb hydra-core ogb
```

## Running the code

1. Your current directory should be `graph_self_supervised_learning`. 
To do that you should run the command:
```commandline
cd ~/path_to_dir/graph_self_supervised_learning
```

3. Run one single experiment to make sure that everything works. 
Provided is a command that runs one self supervised experiment
using the `pubmed` dataset with 1 GCN layer:
```commandline
python run_exp.py -cn reconstruction_agg layer=gcn dataset=pubmed model.depth=1 training.use_self_in_loss=true training.add_regularization=false
```


## Run tensor board

```commandline
tensorboard --logdir=lightning_logs/
```

## Export environment 

```commandline
conda list --explicit > explicit_requirements.txt
conda list > requirements.txt
```