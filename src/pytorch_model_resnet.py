"""
File that implements a model that preforms addition
"""

import torch
from hyper_parameters import Parameters

"""
***************************************************************************************************
ResNet
***************************************************************************************************
"""


class ResNet(torch.nn.Module):
    """
    ***********************************************************************************************
    The pytorch model that adds 2 tensors
    ***********************************************************************************************
    """

    def __init__(self, params: Parameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert params.res_net_mode in ["add", "mul"]
        self.res_net_mode = params.res_net_mode

    def forward(self, x, old_x):
        if self.res_net_mode == "add":
            return x + old_x
        else:
            assert x.shape == old_x.shape
            percentage_change = x + torch.ones_like(x)
            return torch.mul(percentage_change, old_x)
