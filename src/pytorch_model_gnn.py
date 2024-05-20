"""
File that implements the GNN part of the classifier (the starting part).
"""

from typing import Callable

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid

from hyper_parameters import Parameters
from pytorch_model_saver import Saver

"""
***************************************************************************************************
GNNEncoder
***************************************************************************************************
"""


class GNNEncoder(torch.nn.Module):
    """
    ***********************************************************************************************
    The pytorch model that hast many layers of GNNs
    ***********************************************************************************************
    """

    @staticmethod
    def get_layer(class_of_gnn: Callable, in_channels: int, out_channels: int,
                  gnn_params: dict[str, any]) -> torch.nn.Module:
        # check that the inputs make sense
        assert class_of_gnn in [torch_geometric.nn.GCNConv, torch_geometric.nn.GATConv,
                                torch_geometric.nn.GATv2Conv, torch_geometric.nn.SAGEConv,
                                torch_geometric.nn.GraphConv]
        if class_of_gnn in [torch_geometric.nn.GATConv, torch_geometric.nn.GATv2Conv]:
            assert "heads" in gnn_params.keys()

        # return the correct GNN
        if class_of_gnn == torch_geometric.nn.GCNConv:
            return torch_geometric.nn.GCNConv(in_channels=in_channels, out_channels=out_channels,
                                              **gnn_params)
        elif class_of_gnn == torch_geometric.nn.GATConv:
            heads = gnn_params["heads"]
            return torch_geometric.nn.GATConv(in_channels=in_channels,
                                              out_channels=out_channels // heads,
                                              **gnn_params)
        elif class_of_gnn == torch_geometric.nn.GATv2Conv:
            heads = gnn_params["heads"]
            return torch_geometric.nn.GATv2Conv(in_channels=in_channels,
                                                out_channels=out_channels // heads,
                                                **gnn_params)
        elif class_of_gnn == torch_geometric.nn.SAGEConv:
            return torch_geometric.nn.SAGEConv(in_channels=in_channels,
                                               out_channels=out_channels,
                                               **gnn_params)
        elif class_of_gnn == torch_geometric.nn.GraphConv:
            return torch_geometric.nn.GraphConv(in_channels=in_channels, out_channels=out_channels,
                                                **gnn_params)
        else:
            assert False

    """
    ***********************************************************************************************
    The pytorch model that hast many layers of GNNs
    ***********************************************************************************************
    """

    def __init__(self, params: Parameters, list_to_save_to: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = params.in_features
        out_channels = params.hidden_dim
        assert in_channels > 0
        assert out_channels > 0
        assert params.depth > 0
        assert params.use_batch_normalization in [True, False]
        assert params.class_of_gnn in [torch_geometric.nn.GCNConv, torch_geometric.nn.GATConv,
                                       torch_geometric.nn.GATv2Conv, torch_geometric.nn.SAGEConv,
                                       torch_geometric.nn.GraphConv]
        assert params.class_of_activation in [torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.ReLU]

        self.list_to_save_to = list_to_save_to
        layers = [
            (Saver(list_to_save_to=self.list_to_save_to), 'x -> x')
        ]

        previous_output_channels = in_channels
        for d in range(params.depth):
            # add GNN layer
            gnn = self.get_layer(class_of_gnn=params.class_of_gnn,
                                 in_channels=previous_output_channels,
                                 out_channels=out_channels, gnn_params=params.gnn_params)
            layers.append((gnn, 'x, edge_index -> x'))

            # add batch norm layer
            if params.use_batch_normalization:
                layers.append(torch.nn.BatchNorm1d(out_channels))

            # add activation function
            activation_layer = params.class_of_activation(inplace=True)
            layers.append(activation_layer)

            # add layer to save output
            layers.append(Saver(list_to_save_to=self.list_to_save_to))

            previous_output_channels = out_channels

        self.model = torch_geometric.nn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.model.forward(x, edge_index)


"""
***************************************************************************************************
Test
***************************************************************************************************
"""

test_model = """GNNEncoder(
  (model): Sequential(
    (0) - Saver(): x -> x
    (1) - GCNConv(1433, 32): x, edge_index -> x
    (2) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (3) - ELU(alpha=1.0, inplace=True): x -> x
    (4) - Saver(): x -> x
    (5) - GCNConv(32, 32): x, edge_index -> x
    (6) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (7) - ELU(alpha=1.0, inplace=True): x -> x
    (8) - Saver(): x -> x
    (9) - GCNConv(32, 32): x, edge_index -> x
    (10) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (11) - ELU(alpha=1.0, inplace=True): x -> x
    (12) - Saver(): x -> x
    (13) - GCNConv(32, 32): x, edge_index -> x
    (14) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (15) - ELU(alpha=1.0, inplace=True): x -> x
    (16) - Saver(): x -> x
  )
)"""


def test():
    # make dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    assert dataset.num_classes == 7
    assert dataset.num_node_features == 1433
    data = dataset[0]
    assert data.is_undirected()
    assert data.train_mask.sum().item() == 140
    assert data.val_mask.sum().item() == 500
    assert data.test_mask.sum().item() == 1000

    # make gnn encoder
    params = Parameters(in_features=1433, out_features=7)
    saved = []
    gnn_encoder = GNNEncoder(params, list_to_save_to=saved)

    # print model
    print(gnn_encoder)
    assert str(gnn_encoder) == test_model
    gnn_encoder.forward(data)
    assert len(saved) == 5
    gnn_encoder.forward(data)
    assert len(saved) == 10
    assert torch.allclose(saved[-1], gnn_encoder.forward(data))

    # start training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(gnn_encoder.parameters(), lr=0.01, weight_decay=5e-4)

    gnn_encoder.train()
    loss = 1000
    for epoch in range(200):
        optimizer.zero_grad()
        out = F.log_softmax(gnn_encoder(data), dim=1)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"loss = {loss}")
    print(f"final loss = {loss}")

    gnn_encoder.eval()
    pred = F.log_softmax(gnn_encoder(data), dim=1).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')


"""
***************************************************************************************************
call test
***************************************************************************************************
"""

if __name__ == "__main__":
    test()
