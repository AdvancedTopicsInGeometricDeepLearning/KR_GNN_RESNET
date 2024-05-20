"""
File that implements the GNN part of the classifier (the starting part).
"""

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from typing import Callable
from hyper_parameters import Parameters

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

    def __init__(self, params: Parameters, *args, **kwargs):
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

        layers = []

        previous_output_channels = in_channels
        for d in range(params.depth):
            # add GNN layer
            gnn = self.get_layer(class_of_gnn=params.class_of_gnn, in_channels=previous_output_channels,
                                 out_channels=out_channels, gnn_params=params.gnn_params)
            layers.append((gnn, 'x, edge_index -> x'))

            # add batch norm layer
            if params.use_batch_normalization:
                layers.append(torch.nn.BatchNorm1d(out_channels))

            # add activation function
            activation_layer = params.class_of_activation(inplace=True)
            layers.append(activation_layer)

            previous_output_channels = out_channels

        self.model = torch_geometric.nn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.model.forward(x, edge_index)

    def verbose_forward(self, data):
        x, edge_index = data.x, data.edge_index
        result = [x]
        for module in self.model:
            assert not (hasattr(module, 'verbose_forward') and callable(module.verbose_forward))
            assert hasattr(module, 'forward')
            if type(module) in [
                torch_geometric.nn.GCNConv,
                torch_geometric.nn.GATConv,
                torch_geometric.nn.GATv2Conv,
                torch_geometric.nn.SAGEConv,
                torch_geometric.nn.GraphConv
            ]:
                x = module.forward(x, edge_index)
            else:
                x = module.forward(x)
            result.append(x)
        return result


"""
***************************************************************************************************
Test
***************************************************************************************************
"""

test_model = """GNNEncoder(
  (model): Sequential(
    (0) - GCNConv(1433, 32): x, edge_index -> x
    (1) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (2) - ELU(alpha=1.0, inplace=True): x -> x
    (3) - GCNConv(32, 32): x, edge_index -> x
    (4) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (5) - ELU(alpha=1.0, inplace=True): x -> x
    (6) - GCNConv(32, 32): x, edge_index -> x
    (7) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (8) - ELU(alpha=1.0, inplace=True): x -> x
    (9) - GCNConv(32, 32): x, edge_index -> x
    (10) - BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): x -> x
    (11) - ELU(alpha=1.0, inplace=True): x -> x
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
    gnn_encoder = GNNEncoder(params)

    # print model
    print(gnn_encoder)
    assert str(gnn_encoder) == test_model
    assert len(gnn_encoder.verbose_forward(data)) == 13
    assert torch.allclose(gnn_encoder.verbose_forward(data)[-1], gnn_encoder.forward(data))

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
