"""
File that implements the GNN part of the classifier (the starting part).
"""

import torch

"""
***************************************************************************************************
GNNEncoder
***************************************************************************************************
"""


class Saver(torch.nn.Module):
    """
    ***********************************************************************************************
    The pytorch model saves the intermediate results in forward
    ***********************************************************************************************
    """

    def __init__(self, list_to_save_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_to_save_to = list_to_save_to

    def forward(self, data):
        self.list_to_save_to.append(data)
        return data


"""
***************************************************************************************************
Test
***************************************************************************************************
"""


def test():
    # make dataset
    x = torch.ones((4, 4))
    layers = []
    model = Saver(list_to_save_to=layers)
    print(model)
    y = model(x)
    assert torch.allclose(y, x)
    assert len(layers) == 1
    y2 = model(x)
    assert torch.allclose(y2, x)
    assert len(layers) == 2
    layers.clear()
    assert len(layers) == 0
    y = model(x)
    assert torch.allclose(y, x)
    assert len(layers) == 1


"""
***************************************************************************************************
call test
***************************************************************************************************
"""

if __name__ == "__main__":
    test()
