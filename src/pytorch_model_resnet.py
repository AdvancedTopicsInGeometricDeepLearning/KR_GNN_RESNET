"""
File that implements a model that preforms addition
"""

import torch

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(x, y):
        return x + y
