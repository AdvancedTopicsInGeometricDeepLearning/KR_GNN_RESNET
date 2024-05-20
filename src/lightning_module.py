"""
This file implements Pytorch lightning module to make training of pytorch models easier
"""
from typing import Optional

import lightning as L
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from gaussian_kernel import GaussianKernel
from hyper_parameters import Parameters
from pytorch_model_classifier import GNNNodeClassifier

"""
***************************************************************************************************
PytorchLightningModule
***************************************************************************************************
"""


class PytorchLightningModuleNodeClassifier(L.LightningModule):
    """
    ***********************************************************************************************
    Pytorch Lightning Module To make training and testing easier
    ***********************************************************************************************
    """

    def __init__(
            self,
            params: Parameters
    ):
        super().__init__()
        self.model = GNNNodeClassifier(params=params)
        if params.use_kernel_regression:
            kernel = GaussianKernel(
                max_samples=4096,
                add_regularization=params.add_regularization_to_kernel_regression
            )
            self.kernel_regression_loss = lambda x, y: kernel.compute_d(x, y)

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
