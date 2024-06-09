"""
This file implements Pytorch lightning module to make training of pytorch models easier
"""
from typing import Optional, Any

import lightning as L
import torch
import torch.nn.functional as F
import torch_geometric.data
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from gaussian_kernel import GaussianKernel
from hyper_parameters import Parameters, KernelRegressionMode
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
        self.params = params
        self.train_epoch_count = 0
        self.kernel_regression_loss = None
        if params.kernel_regression_mode != KernelRegressionMode.OFF:
            kernel = GaussianKernel(
                max_samples=4096,
                add_regularization=params.add_regularization_to_kernel_regression
            )
            self.kernel_regression_loss = lambda x, y: kernel.compute_d(x, y)

    """
    ***********************************************************************************************
    for calculating KR
    ***********************************************************************************************
    """

    def get_kernel_regression_loss(
            self,
            kr_checkpoints: list[torch.Tensor],
            mask: torch.Tensor,
            data: torch_geometric.data.Data
    ):
        assert self.kernel_regression_loss
        kr_loss = torch.zeros(())
        y = data.y[mask].type(torch.FloatTensor)
        for layer_out in kr_checkpoints:
            layer_out = layer_out[mask]
            kr_loss += self.kernel_regression_loss(x=layer_out, y=y)
        return kr_loss * self.params.kernel_regression_loss_lambda

    """
    ***********************************************************************************************
    API for pytorch lightning
    ***********************************************************************************************
    """

    def forward(self, data, mode="train", allow_kr=True) -> tuple[torch.Tensor, torch.Tensor]:
        # x, edge_index = data.x, data.edge_index
        if self.kernel_regression_loss and mode == "train":
            (x, kr_checkpoints) = self.model.verbose_forward(data)
        else:
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

        # add KR loss if applicable
        if self.kernel_regression_loss and mode == "train" and allow_kr:
            loss += self.get_kernel_regression_loss(kr_checkpoints=kr_checkpoints, mask=mask,
                                                    data=data)

        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, acc = self.forward(batch, mode="train")
        self.log("training loss", loss)
        self.log("training accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val", allow_kr=False)
        self.log("validation loss", loss)
        self.log("validation accuracy", acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="test", allow_kr=False)
        self.log("test loss", loss)
        self.log("test accuracy", acc)

    def on_train_epoch_end(self) -> None:
        self.train_epoch_count += 1

    def log(self, *args: Any, **kwargs: Any) -> None:
        super().log(*args, **kwargs, batch_size=self.params.batch_size)
