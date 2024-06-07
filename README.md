# KR_GNN_RESNET
A repo that presents our experiments with using kernel
regression with graph neural networks while utilizing skip 
connections. This project houses the results for the paper.

To run all the experiments that we ran simply run this command 
(assuming you have all the dependencies)

```commandline
python src/run_main.py
```

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