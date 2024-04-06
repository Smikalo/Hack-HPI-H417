from torch_geometric.datasets import Planetoid
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# to install the right dependencies
#import torch
#print(f'torch version: {torch.__version__}')
#os.environ['TORCH'] = torch.__version__
# note used:
#!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
#!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html

import torch.nn as TNN
from torch_geometric.nn import GCNConv
from torch.optim import Adam as TOadam
from torch import manual_seed


def main(datasetname='Cora', summarize_data=True, try_mlp=False, try_gcn=False, compare_mlp_gcn=True, epochcount=201):
    """
    This script takes a dataset from torch_geometric,
    Trains and compares simple (single hidden layer) MLP and GCN models
    """

    dataset = load_data(datasetname)

    if summarize_data:
        # summarize the first graph object.
        summarize_data(dataset[0])

    # define an MLP model
    model_mlp = MLP(dataset.num_features,  # c_in
                    dataset.num_classes,  # c_out
                    16)  # c_hidden

    # define a GCN model
    model_gcn = GCN(dataset.num_features,  # c_in
                    dataset.num_classes,  # c_out
                    16)  # c_hidden

    if try_mlp:
        try_model(model_mlp, 'MLP', dataset[0], epochcount)

    if try_gcn:
        try_model(model_gcn, 'GCN', dataset[0], epochcount)

    if compare_mlp_gcn:
        if not try_mlp:
            try_model(model_mlp, 'MLP',
                      dataset[0], epochcount, visualize=False)
        if not try_gcn:
            try_model(model_gcn, 'GCN',
                      dataset[0], epochcount, visualize=False)
        compare_models([model_mlp, model_gcn], ['MLP', 'GCN'], dataset[0])


def compare_models(models, modelnames, data):
    # Visualize the 7-dimensional node embeddings that our untrained model produces, using the TSNE package that projects a high-dimensional data onto a low-dimensional plane in such a way that similar points are put close to each other.
    print(f'Comparing: ', end='')
    outs = [None] * len(modelnames)
    titles = [None] * len(modelnames)
    for i, modelname in enumerate(modelnames):
        print(modelname, end=' ')
        model = models[i]
        model.eval()
        if modelname in ['MLP']:
            out = model(data.x)
        elif modelname in ['GCN']:
            out = model(data.x, data.edge_index)
        outs[i] = out
        titles[i] = f'Trained {modelname} embeddings'

    f_modelnames = '_'.join(modelnames)
    visualize_both(outs[0], outs[1], titles[0], titles[1],
                   color=data.y, fname=f'Cora_scatter_{f_modelnames}_trained.png')


def try_model(model, modelname, data, epochcount, visualize=True):
    print(model)
    if visualize:
        vis_model(model, data, modelname, 'untrained')
    train_test(model, modelname, data, epochcount)
    if visualize:
        vis_model(model, data, modelname, 'trained')


def train_test(model, modelname, data, epochcount):
    print(f'Training and testing: {modelname}')

    loss_function = TNN.CrossEntropyLoss()
    optimizer = TOadam(model.parameters(),
                       lr=0.01,
                       weight_decay=5e-4)

    for epoch in range(1, epochcount+1):
        loss = train(model, modelname, optimizer, data, loss_function)
        train_acc = test(model, modelname, data, data.train_mask)
        val_acc = test(model, modelname, data, data.val_mask)
        test_acc = test(model, modelname, data, data.test_mask)

        print(
            f"Epoch: {epoch:03d}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}"
        )


def vis_model(model, data, modelname, suffix):
    # Visualize the 7-dimensional node embeddings that our untrained model produces, using the TSNE package that projects a high-dimensional data onto a low-dimensional plane in such a way that similar points are put close to each other.
    print(f'Visualizing: {modelname}')
    model.eval()
    if modelname in ['MLP']:
        out = model(data.x)
    elif modelname in ['GCN']:
        out = model(data.x, data.edge_index)
    visualize(out, color=data.y,
              fname=f'Cora_scatter_{modelname}_{suffix}.png')


def train(model, modelname, optimizer, data, loss_function):
    model.train()
    optimizer.zero_grad()
    if modelname in ['MLP']:
        out = model(data.x)
    elif modelname in ['GCN']:
        out = model(data.x, data.edge_index)
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # compute grads
    optimizer.step()  # apply grads
    return loss


def test(model, modelname, data, mask):
    model.eval()
    if modelname in ['MLP']:
        out = model(data.x)
    elif modelname in ['GCN']:
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc


class GCN(TNN.Module):
    def __init__(self, c_in, c_out, c_hidden):
        super().__init__()
        manual_seed(12345)
        self.conv1 = GCNConv(c_in, c_hidden)
        self.conv2 = GCNConv(c_hidden, c_out)
        self.dropout = TNN.Dropout(p=0.5, inplace=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)  # Output dim = c_hidden
        x = TNN.ReLU()(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)  # Input dim = c_hidden


class MLP(TNN.Module):
    def __init__(self, c_in, c_out, c_hidden):
        super().__init__()
        manual_seed(12345)
        self.lin1 = TNN.Linear(c_in, c_hidden)
        self.lin2 = TNN.Linear(c_hidden, c_out)
        self.dropout = TNN.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        x = self.lin1(x)  # output dim = c_hidden
        x = TNN.ReLU()(x)
        x = self.dropout(x)  # input dim = c_hidden
        return self.lin2(x)


def load_data(datasetname):
    dataset = Planetoid(root='data/Planetoid', name=datasetname)
    print(f'Dataset: {dataset}:')
    print('=========================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    return(dataset)


def summarize_data(data):
    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {(2 * data.num_edges) / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(
        f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def visualize(h, color, fname):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    # plt.show()
    plt.savefig(fname)


def visualize_both(h1, h2, title1, title2, color, fname):
    z1 = TSNE(n_components=2).fit_transform(h1.detach().cpu().numpy())
    z2 = TSNE(n_components=2).fit_transform(h2.detach().cpu().numpy())

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].scatter(z1[:, 0], z1[:, 1], s=70, c=color, cmap="Set2")
    ax[0].set_title(title1)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].scatter(z2[:, 0], z2[:, 1], s=70, c=color, cmap="Set2")
    ax[1].set_title(title2)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # plt.show()
    plt.savefig(fname)


if __name__ == "__main__":
    main()
