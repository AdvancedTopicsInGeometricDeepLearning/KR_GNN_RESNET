"""
File that contains all the possible parameter to the lightning module
"""

from enum import Enum
from typing import Callable

import torch
import torch_geometric.data.data

"""
***************************************************************************************************
types
***************************************************************************************************
"""


class KernelRegressionMode(Enum):
    OFF = 1
    BEFORE_SKIP_CONNECTION = 2
    AFTER_SKIP_CONNECTION = 3
    AFTER_EACH_BLOCK = 4


class ResNetMode(Enum):
    NONE = 1
    ADD = 2
    MUL = 3


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
    kernel_regression_mode: KernelRegressionMode
    add_regularization_to_kernel_regression: bool
    max_edges_for_kr_loss: int
    kernel_regression_loss_lambda: float
    batch_size: int
    max_epochs: int
    min_epochs: int
    early_stopping_patience: int
    learning_rate: float
    skip_connection_stride: int
    res_net_mode: ResNetMode

    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_regression_mode: KernelRegressionMode,
            depth: int,
            res_net_mode: ResNetMode
    ):
        self.in_features = in_features
        self.hidden_dim = 32
        self.out_features = out_features
        self.depth = depth
        self.use_batch_normalization = True
        self.gnn_params = {}
        self.class_of_gnn = torch_geometric.nn.GCNConv
        self.class_of_activation = torch.nn.ELU
        self.kernel_regression_mode = kernel_regression_mode
        self.add_regularization_to_kernel_regression = True
        self.max_edges_for_kr_loss = 10000
        self.kernel_regression_loss_lambda = 1.0
        self.batch_size = 1
        self.max_epochs = 100
        self.min_epochs = 30
        self.early_stopping_patience = 3
        self.learning_rate = 1e-2
        self.skip_connection_stride = 2
        self.res_net_mode = res_net_mode
