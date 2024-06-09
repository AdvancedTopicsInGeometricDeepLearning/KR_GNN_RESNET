# KR_GNN_RESNET
A repo that presents our experiments with using kernel
regression with graph neural networks while utilizing skip 
connections. This project houses the results for the paper.

To run all the experiments that we ran simply run this command 
(assuming you have all the dependencies)

```commandline
python src/run_main.py
```

Make sure that the current working directory is **not** inside of `src`.


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

## Run tensor board

If you would like to view the training progress during training you can use tensorboard:

```commandline
tensorboard --logdir=lightning_logs/
```

## Export environment 

To export the environment to allow for the re-creation of the tests: 

```commandline
conda list --explicit > explicit_requirements.txt
conda list > requirements.txt
```