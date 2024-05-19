"""
main file that runs an experiment on
"""

import lightning as L
import torch
import torch_geometric.data.data
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch_geometric.datasets import Planetoid

from lightning_module import PytorchLightningModuleNodeClassifier

"""
***************************************************************************************************
helper functions
***************************************************************************************************
"""


def run_experiment():
    # get dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    # make data loader
    node_data_loader = torch_geometric.data.DataLoader(dataset, batch_size=1)

    # make model
    model = PytorchLightningModuleNodeClassifier(
        in_features=dataset.num_node_features,
        hidden_dim=32,
        out_features=dataset.num_classes,
        depth=40,
        use_batch_normalization=True,
        class_of_gnn=torch_geometric.nn.GCNConv, gnn_params={},
        class_of_activation=torch.nn.ELU
    )

    # make trainer
    trainer = L.Trainer(
        max_epochs=1000,
        callbacks=[EarlyStopping(monitor="validation loss", mode="min")]
    )

    # train model
    trainer.fit(model=model, train_dataloaders=node_data_loader, val_dataloaders=node_data_loader)

    # test model to get accuracy
    acc = trainer.test(model=model, dataloaders=node_data_loader)

    return acc[0]


"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    acc1 = run_experiment()
    print(f"acc1 = {acc1}")


"""
***************************************************************************************************
run main
***************************************************************************************************
"""

if __name__ == "__main__":
    main()
