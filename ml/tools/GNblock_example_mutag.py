import os
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean


def main():
    dataset = TUDataset(root='data/TUDataset',
                        name='MUTAG',
                        transform=add_global_attr)
    summarize_dataset(dataset)

    train_loader, test_loader = get_data_loaders(dataset)

    # build the model:
    edge_model = EdgeModel(64, dataset)
    node_model = NodeModel(64, dataset)
    global_model = GlobalModel(64, dataset)
    NUM_PASSES = 2

    gn_model = GN(edge_model, node_model, global_model, dataset, NUM_PASSES)
    optimizer = torch.optim.Adam(gn_model.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss()

    print(gn_model)

    for epoch in range(1, 120):
        train(gn_model, train_loader, loss_function, optimizer)
        train_acc = test(gn_model, train_loader)
        test_acc = test(gn_model, test_loader)
        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


def train(gn_model, train_loader, loss_function, optimizer):
    gn_model.train()

    for data in train_loader:  # Iterate over the batches of grahs
        out = gn_model(data.x, data.edge_attr, data.u,
                       data.edge_index, data.batch)  # Forward pass(es)
        loss = loss_function(out, data.y)  # Compute the loss
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the weights based on the computed gradients
        optimizer.zero_grad()  # Clear the computed gradients


def test(gn_model, loader):
    gn_model.eval()

    correct = 0
    for data in loader:
        # Iterate over the batches
        out = gn_model(data.x, data.edge_attr, data.u,
                       data.edge_index, data.batch)
        # Predict the labels using the label with the highest probability
        pred = out.argmax(dim=1)
        # Check against the ground truth
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)  # Compute accuracy


class GN(torch.nn.Module):
    def __init__(self, edge_model, node_model, global_model, dataset, num_passes):
        super().__init__()
        torch.manual_seed(12345)
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.lin = torch.nn.Linear(num_features, dataset.num_classes)
        self.num_passes = num_passes
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, node_attr, edge_attr, u, edge_index, batch):
        # these arguments are provided when GN is called:
        #node_attr = data.x
        #edge_attr = data.edge_attr
        #u = data.u
        #edge_index = data.edge_index
        # 1. perform mesasge passing to obtain graph embeddings
        for _ in range(self.num_passes):
            src_node_idx = edge_index[0]
            dest_node_idx = edge_index[1]
            edge_attr = self.edge_model(
                node_attr[src_node_idx],
                node_attr[dest_node_idx],
                edge_attr,
                u,
                # to convert batch to the dim E = num of edges
                batch[src_node_idx]
            )

            node_attr, edge_out_bar = self.node_model(
                node_attr,
                edge_index,
                edge_attr,
                u,
                batch
            )

            u = self.global_model(node_attr,
                                  edge_out_bar,
                                  u,
                                  batch
                                  )

        # 2. Readout layer
        graph_attr = torch.cat([node_attr, edge_out_bar, u[batch]], dim=1)
        graph_attr = global_mean_pool(graph_attr, batch)

        return self.lin(graph_attr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')


class EdgeModel(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(12345)
        num_features = dataset.num_features * 2 + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, dataset.num_edge_features),
            torch.nn.ReLU()
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(12345)
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, dataset.num_features),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes in all graphs of the batch.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        dest_node_idx = edge_index[1]
        # Average all attr of incoming edges for each dest node
        edge_out_bar = scatter_mean(src=edge_attr, index=dest_node_idx, dim=0)
        out = torch.cat([x, edge_out_bar, u[batch]], 1)
        return self.node_mlp(out), edge_out_bar


class GlobalModel(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(12345)
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.ReLU()
        )

    def forward(self, node_attr_prime, edge_out_bar, u, batch):
        # node_attr_bar: [N, F_x], where N is the number of nodes in the batch.
        # edge_attr: [N, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # Average all node attributes for each graph, using batch tensor.
        node_attr_bar = scatter_mean(node_attr_prime, batch, 0)
        # Average all edge attributes for each graph, using batch
        edge_attr_bar = scatter_mean(edge_out_bar, batch, 0)
        out = torch.cat([u, node_attr_bar, edge_attr_bar], dim=1)
        return self.global_mlp(out)


def get_data_loaders(dataset):
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print('Graphs in the training set')
    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data.batch)
        print()

    return train_loader, test_loader


def add_global_attr(data):
    data.u = torch.tensor([[0]]).to(torch.float32)
    return data


def summarize_dataset(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Have a look at the first graph.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


if __name__ == '__main__':
    main()
