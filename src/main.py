"""
main file that runs an experiment on
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch_geometric.data.data
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid

"""
***************************************************************************************************
helper functions
***************************************************************************************************
"""


def run(
        dataset,
        model,
        str_optimizer,
        runs,
        epochs,
        lr,
        weight_decay,
        early_stopping,
        logger,
        momentum,
        eps,
        update_freq,
        gamma,
        alpha,
        hyperparam
):
    if logger is not None:
        if hyperparam:
            logger += f"-{hyperparam}{eval(hyperparam)}"
        path_logger = os.path.join(path_runs, logger)
        print(f"path logger: {path_logger}")

        ut.empty_dir(path_logger)
        logger = SummaryWriter(
            log_dir=os.path.join(path_runs, logger)) if logger is not None else None

    val_losses, accs, durations = [], [], []
    torch.manual_seed(42)
    for i_run in range(runs):
        data = dataset[0]
        data = data.to(device)

        model.to(device).reset_parameters()

        if str_optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif str_optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            lam = (float(epoch) / float(epochs)) ** gamma if gamma is not None else 0.
            train(model, optimizer, data, preconditioner, lam)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = int(epoch)
            eval_info['run'] = int(i_run + 1)
            eval_info['time'] = time.perf_counter() - t_start
            eval_info['eps'] = eps
            eval_info['update-freq'] = update_freq

            if gamma is not None:
                eval_info['gamma'] = gamma

            if alpha is not None:
                eval_info['alpha'] = alpha

            if logger is not None:
                for k, v in eval_info.items():
                    logger.add_scalar(k, v, global_step=epoch)

            if eval_info['val loss'] < best_val_loss:
                best_val_loss = eval_info['val loss']
                test_acc = eval_info['test acc']

            val_loss_history.append(eval_info['val loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val loss'] > tmp.mean().item():
                    break
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    if logger is not None:
        logger.close()
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print('Val Loss: {:.4f}, Test Accuracy: {:.2f} ± {:.2f}, Duration: {:.3f} \n'.
          format(loss.mean().item(),
                 100 * acc.mean().item(),
                 100 * acc.std().item(),
                 duration.mean().item()))


def train(model, optimizer, data, lam=0.):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False

    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])

    loss.backward(retain_graph=True)
    optimizer.step()


def evaluate(model: torch.nn.Module, data: torch_geometric.data.data.Data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc

    return outs


"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--logger', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--preconditioner', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--update_freq', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--hyperparam', type=str, default=None)

    # get dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')


"""
***************************************************************************************************
run main
***************************************************************************************************
"""

if __name__ == "__main__":
    main()
