"""
This file implements Pytorch lightning module to make training of pytorch models easier
"""
from typing import Optional

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

    def calculate_kernel_regression_loss(
            self, edge_index: torch.Tensor, out: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = [None] * len(out)
        global_loss = torch.zeros(())
        for i in range(1, len(out) - 1):
            relevant_edges = edge_index.T
            if relevant_edges.shape[0] == 0:
                loss[i] = None
                continue
            elif relevant_edges.shape[0] > self.params.max_edges_for_kr_loss:
                idx_to_take = torch.randperm(relevant_edges.shape[0])[
                              :self.cfg.training.max_edges_for_loss
                              ]
                relevant_edges = relevant_edges[idx_to_take]
            neighbours_emb = out[i - 1]
            target_emb = out[i]
            source_nodes, target_nodes = torch.split(relevant_edges, 1, dim=1)
            source_nodes = source_nodes.flatten()
            target_nodes = target_nodes.flatten()

            # Need to detach the neighbours from the loss calculation
            neighbours_emb = neighbours_emb.clone()

            if neighbours_emb.requires_grad:
                neighbours_emb.register_hook(lambda grad: torch.zeros_like(grad))
            selected_neighbours = neighbours_emb[source_nodes]

            selected_targets = target_emb[target_nodes]

            lvl_loss = self.kernel_regression_loss(x=selected_targets, y=selected_neighbours)
            if self.params.use_self_in_loss_for_kernel_regression:
                lvl_loss += self.kernel_regression_loss(x=selected_targets,
                                                        y=neighbours_emb[target_nodes])
            if torch.any(torch.isnan(lvl_loss)).item():
                print("Warning: got nan in loss computation")
                loss[i] = None
                continue
            self.log(f"KR loss of level {i}", lvl_loss.item())
            loss[i] = lvl_loss
            global_loss += lvl_loss
        return global_loss

    def add_kernel_regression_loss(
            self,
            loss: torch.Tensor,
            layers: list[torch.Tensor],
            mask: torch.Tensor,
            data: torch_geometric.data.Data
    ):
        assert self.kernel_regression_loss
        criterion_mask = mask
        root_nodes_idx = torch.where(criterion_mask)[0]
        relevant_edges = sum(data.edge_index[1] == i for i in root_nodes_idx).bool()
        assert len(relevant_edges) != 0, "Warning got graph without edges in embedding phase"
        relevant_nodes = torch.cat([data.edge_index[0][relevant_edges], root_nodes_idx], dim=0)
        relevant_nodes = torch.unique(relevant_nodes)
        edge_index = torch_geometric.utils.subgraph(relevant_nodes, data.edge_index,
                                                    num_nodes=data.x.size(0))[0]
        try:
            added_loss = self.calculate_kernel_regression_loss(edge_index=edge_index, out=layers)
            loss += added_loss
        except RuntimeError as re:
            return

    """
    ***********************************************************************************************
    API for pytorch lightning
    ***********************************************************************************************
    """

    def forward(self, data, mode="train"):
        # x, edge_index = data.x, data.edge_index
        if self.kernel_regression_loss and mode == "train":
            layers = self.model.verbose_forward(data)
            x = layers[-1]
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
        if self.kernel_regression_loss and mode == "train":
            self.add_kernel_regression_loss(loss=loss, layers=layers, mask=mask, data=data)

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
        loss, acc = self.forward(batch, mode="val")
        self.log("validation loss", loss)
        self.log("validation accuracy", acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="test")
        self.log("test loss", loss)
        self.log("test accuracy", acc)

    def on_train_epoch_end(self) -> None:
        self.train_epoch_count += 1
