"""
File that implements a classifier for nodes in a graph using a deep GNN layer and a linear layer.
"""

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid

from pytorch_model_gnn import GNNEncoder

"""
***************************************************************************************************
GNNNodeClassifier
***************************************************************************************************
"""


class GNNNodeClassifier(torch.nn.Module):
    """
    ***********************************************************************************************
    The pytorch model that hast many layers of GNNs then a linear layer.
    This model classifies node in the graph using logit encodings.
    ***********************************************************************************************
    """

    def __init__(self, in_features: int, hidden_dim: int, out_features: int, depth: int,
                 use_batch_normalization: bool,
                 class_of_gnn, gnn_params: dict[str, any], class_of_activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert out_features > 0
        # add encoder
        encoder = GNNEncoder(in_channels=in_features, out_channels=hidden_dim, depth=depth,
                             use_batch_normalization=use_batch_normalization,
                             class_of_gnn=class_of_gnn, gnn_params=gnn_params,
                             class_of_activation=class_of_activation)

        self.model = torch.nn.Sequential(
            encoder,
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            class_of_activation(),
            torch.nn.Linear(in_features=hidden_dim, out_features=out_features),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, data):
        return self.model.forward(data)

    def verbose_forward(self, data):
        result = [data]
        # go over the layers in the network
        for module in self.model:
            if hasattr(module, 'verbose_forward') and callable(module.verbose_forward):
                # if the layer has the same function then call that
                r = module.verbose_forward(data)
                data = r[-1]
                result += r
            else:
                data = module(data)
        result.append(data)
        return result


"""
***************************************************************************************************
Test
***************************************************************************************************
"""

test_model = """GNNNodeClassifier(
  (model): Sequential(
    (0): GNNEncoder(
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
    )
    (1): Linear(in_features=32, out_features=32, bias=True)
    (2): ELU(alpha=1.0)
    (3): Linear(in_features=32, out_features=7, bias=True)
    (4): LogSoftmax(dim=1)
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
    classifier = GNNNodeClassifier(in_features=1433, hidden_dim=32, out_features=7, depth=4,
                                    use_batch_normalization=True,
                                    class_of_gnn=torch_geometric.nn.GCNConv, gnn_params={},
                                    class_of_activation=torch.nn.ELU)

    # print model
    print(classifier)
    assert str(classifier) == test_model
    verbose_out = classifier.verbose_forward(data)
    assert len(verbose_out) == 6

    # start training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=5e-4)

    classifier.train()
    loss = 1000
    for epoch in range(200):
        optimizer.zero_grad()
        out = classifier(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"loss = {loss}")

    assert loss < 0.0004

    classifier.eval()
    pred = classifier(data).argmax(dim=1)
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
