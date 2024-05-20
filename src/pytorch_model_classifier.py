"""
File that implements a classifier for nodes in a graph using a deep GNN layer and a linear layer.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from hyper_parameters import Parameters
from pytorch_model_gnn import GNNEncoder
from pytorch_model_saver import Saver

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

    def __init__(self, params: Parameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert params.out_features > 0
        # add encoder
        self.list_to_save_to = []
        encoder = GNNEncoder(params=params, list_to_save_to=self.list_to_save_to)

        self.model = torch.nn.Sequential(
            encoder,
            torch.nn.Linear(in_features=params.hidden_dim, out_features=params.hidden_dim),
            params.class_of_activation(),
            torch.nn.Linear(in_features=params.hidden_dim, out_features=params.out_features),
            torch.nn.LogSoftmax(dim=1),
            Saver(list_to_save_to=self.list_to_save_to)
        )

    def forward(self, data):
        x = self.model.forward(data)
        self.list_to_save_to.clear()
        return x

    def verbose_forward(self, data):
        self.list_to_save_to.clear()
        self.model.forward(data)
        return self.list_to_save_to


"""
***************************************************************************************************
Test
***************************************************************************************************
"""

test_model = """GNNNodeClassifier(
  (model): Sequential(
    (0): GNNEncoder(
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
    )
    (1): Linear(in_features=32, out_features=32, bias=True)
    (2): ELU(alpha=1.0)
    (3): Linear(in_features=32, out_features=7, bias=True)
    (4): LogSoftmax(dim=1)
    (5): Saver()
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
    classifier = GNNNodeClassifier(params=params)

    # print model
    print(classifier)
    assert str(classifier) == test_model
    verbose_out = classifier.verbose_forward(data)
    print(len(verbose_out))
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
