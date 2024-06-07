"""
main file that runs a single experiment
"""
import argparse
import multiprocessing

import lightning as L
import torch_geometric
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch_geometric.datasets import Planetoid

from hyper_parameters import Parameters, KernelRegressionMode, ResNetMode
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
        # Sample 30 neighbors for each node for all iterations
        num_neighbors=[30] * params.depth,
        batch_size=params.batch_size,
        input_nodes=dataset.train_mask if mode == "train" else (
            dataset.val_mask if mode == "val" else dataset.test_mask),
        num_workers=multiprocessing.cpu_count()
    )


def run_experiment(seed: int, depth: int, use_kr: KernelRegressionMode, res_net_mode: ResNetMode):
    L.seed_everything(seed, workers=True)

    # get dataset
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed')

    # Make parameters
    params = Parameters(
        in_features=dataset.num_node_features, out_features=dataset.num_classes, depth=depth,
        kernel_regression_mode=use_kr, res_net_mode=res_net_mode
    )

    # make model
    model = PytorchLightningModuleNodeClassifier(params=params)

    # make trainer
    trainer = L.Trainer(
        max_epochs=params.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="validation accuracy", mode="max",
                patience=params.early_stopping_patience
            )
        ],
        deterministic=True
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

    return {
        "test accuracy": acc[0]['test accuracy'],
        "test loss": acc[0]['test loss'],
        "train epochs": model.train_epoch_count,
        "model": str(model)
    }


"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    parser = argparse.ArgumentParser(
        prog='Experiment Runner',
        description="Runs a single experiment with the provided parameters.",
        epilog='Thanks for using this tool'
    )
    parser.add_argument(
        '--kr', type=str, choices=["off", "before", "after"], required=True,
        help="Choose Kernel regression settings either off, or before/after the skip connection."
    )
    parser.add_argument(
        '--depth', type=int, required=True,
        help="The amount of GNNs to use in the neural network."
    )
    parser.add_argument(
        '--residual', type=str, choices=["none", "add", "mul"], required=True,
        help="The type of skip connection to use."
    )
    args = parser.parse_args()

    use_kr = None
    match args.kr:
        case "off":
            use_kr = KernelRegressionMode.OFF
        case "before":
            use_kr = KernelRegressionMode.BEFORE_SKIP_CONNECTION
        case "after":
            use_kr = KernelRegressionMode.AFTER_SKIP_CONNECTION

    res_net_mode = None
    match args.residual:
        case "none":
            res_net_mode = ResNetMode.NONE
        case "add":
            res_net_mode = ResNetMode.ADD
        case "mul":
            res_net_mode = ResNetMode.MUL

    acc1 = run_experiment(seed=42, depth=args.depth, use_kr=use_kr, res_net_mode=res_net_mode)
    print(f"acc1 = {acc1}")


"""
***************************************************************************************************
run main
***************************************************************************************************
"""

if __name__ == "__main__":
    main()
