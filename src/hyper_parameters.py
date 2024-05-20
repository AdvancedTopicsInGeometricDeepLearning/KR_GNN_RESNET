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
