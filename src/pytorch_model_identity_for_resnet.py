"""
File that implements a model that does nothing at all. This is used for marking the
start of a ResNet block.
"""

import torch

"""
***************************************************************************************************
Identity
***************************************************************************************************
"""


class IdentityForResNet(torch.nn.Module):
    """
    ***********************************************************************************************
    The pytorch model that does nothing
    ***********************************************************************************************
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(data):
        return data
