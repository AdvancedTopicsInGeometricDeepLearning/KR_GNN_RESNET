"""
main file that runs an experiment on
"""

import lightning as L
import torch_geometric.data.data
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch_geometric.datasets import Planetoid

from hyper_parameters import Parameters
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
    params = Parameters(in_features=dataset.num_node_features, out_features=dataset.num_classes)
    model = PytorchLightningModuleNodeClassifier(params=params)

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
