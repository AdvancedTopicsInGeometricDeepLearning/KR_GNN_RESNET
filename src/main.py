"""
main file that runs an experiment on
"""
import multiprocessing

import lightning as L
import torch_geometric
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch_geometric.datasets import Planetoid

from hyper_parameters import Parameters
from lightning_module import PytorchLightningModuleNodeClassifier

"""
***************************************************************************************************
helper functions
***************************************************************************************************
"""


def get_data_loader(
        dataset: torch_geometric.datasets.Planetoid,
        mode: str,
        params: Parameters
) -> torch_geometric.loader.DataLoader:
    assert mode in ["train", "val", "test"]
    return torch_geometric.loader.NeighborLoader(
        # planetoid contains only one graph
        dataset[0],
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * params.depth,
        # Use a batch size of 128 for sampling training nodes
        batch_size=params.batch_size,
        input_nodes=dataset.train_mask if mode == "train" else (
            dataset.val_mask if mode == "val" else dataset.test_mask),
        num_workers=multiprocessing.cpu_count()
    )


def run_experiment():
    # get dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    # Make parameters
    params = Parameters(in_features=dataset.num_node_features, out_features=dataset.num_classes)
    params.use_kernel_regression = True

    # make model
    model = PytorchLightningModuleNodeClassifier(params=params)

    # make trainer
    trainer = L.Trainer(
        max_epochs=params.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="validation loss", mode="min",
                patience=params.early_stopping_patience
            )
        ]
    )

    # train model
    trainer.fit(
        model=model,
        train_dataloaders=get_data_loader(dataset=dataset, mode="train", params=params),
        val_dataloaders=get_data_loader(dataset=dataset, mode="val", params=params)
    )

    # test model to get accuracy
    acc = trainer.test(
        model=model,
        dataloaders=get_data_loader(dataset=dataset, mode="test", params=params)
    )

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
