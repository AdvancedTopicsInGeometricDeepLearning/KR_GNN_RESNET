"""
File that implements a model that preforms addition
"""

import torch

from hyper_parameters import Parameters, ResNetMode

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
        self.res_net_mode = params.res_net_mode

    def forward(self, x, old_x):
        assert x.shape == old_x.shape
        match self.res_net_mode:
            case ResNetMode.ADD:
                return x + old_x
            case ResNetMode.MUL:
                percentage_change = x + torch.ones_like(x)
                return torch.mul(percentage_change, old_x)
            case _:
                assert False
