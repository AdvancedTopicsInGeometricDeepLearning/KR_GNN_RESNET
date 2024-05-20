"""
File that contains all the possible parameter to the lightning module
"""

import torch
import torch_geometric.data.data
from typing import Callable

"""
***************************************************************************************************
class for parameters
***************************************************************************************************
"""


class Parameters:
    in_features: int
    hidden_dim: int
    out_features: int
    depth: int
    use_batch_normalization: bool
    gnn_params: dict[str, any]
    class_of_gnn: Callable
    class_of_activation: Callable
    use_kernel_regression: bool
    add_regularization_to_kernel_regression: bool
    use_self_in_loss_for_kernel_regression: bool
    max_edges_for_kr_loss: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.hidden_dim = 32
        self.out_features = out_features
        self.depth = 4
        self.use_batch_normalization = True
        self.gnn_params = {}
        self.class_of_gnn = torch_geometric.nn.GCNConv
        self.class_of_activation = torch.nn.ELU
        self.use_kernel_regression = False
        self.add_regularization_to_kernel_regression = False
        self.use_self_in_loss_for_kernel_regression = False
        self.max_edges_for_kr_loss = 10000
        self.batch_size = 8
        self.max_epochs = 1000
        self.early_stopping_patience = 10
        self.learning_rate = 1e-3
