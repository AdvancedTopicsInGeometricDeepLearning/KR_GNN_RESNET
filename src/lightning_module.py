"""
This file implements Pytorch lightning module to make training of pytorch models easier
"""
from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
import torch_geometric
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torch_geometric.datasets import Planetoid

from pytorch_model_classifier import GNNNodeClassifier

"""
***************************************************************************************************
PytorchLightningModule
***************************************************************************************************
"""


class PytorchLightningModule(L.LightningModule):
    """
    ***********************************************************************************************
    Pytorch Lightning Module To make training and testing easier
    ***********************************************************************************************
    """

    def __init__(self, in_features: int, hidden_dim: int, out_features: int, depth: int,
                 use_batch_normalization: bool,
                 class_of_gnn, gnn_params: dict[str, any], class_of_activation):
        super().__init__()
        self.model = GNNNodeClassifier(in_features=in_features, hidden_dim=hidden_dim,
                                       out_features=out_features, depth=depth,
                                       use_batch_normalization=use_batch_normalization,
                                       class_of_gnn=class_of_gnn, gnn_params=gnn_params,
                                       class_of_activation=class_of_activation)

    def forward(self, data, mode="train"):
        # x, edge_index = data.x, data.edge_index
        x = self.model.forward(data)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = F.nll_loss(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, acc = self.forward(batch, mode="train")
        self.log("training loss", loss)
        self.log("training accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        self.log("validation loss", loss)
        self.log("validation accuracy", acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="test")
        self.log("test loss", loss)
        self.log("test accuracy", acc)


"""
***************************************************************************************************
test
***************************************************************************************************
"""


def test():
    model = PytorchLightningModule(in_features=1433, hidden_dim=32, out_features=7, depth=4,
                                   use_batch_normalization=True,
                                   class_of_gnn=torch_geometric.nn.GCNConv, gnn_params={},
                                   class_of_activation=torch.nn.ELU)
    trainer = L.Trainer(max_epochs=20)
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    node_data_loader = torch_geometric.data.DataLoader(dataset, batch_size=1)
    trainer.fit(model=model, train_dataloaders=node_data_loader, val_dataloaders=node_data_loader)
    trainer.test(model=model, dataloaders=node_data_loader)


"""
***************************************************************************************************
call test 
***************************************************************************************************
"""
if __name__ == "__main__":
    test()
