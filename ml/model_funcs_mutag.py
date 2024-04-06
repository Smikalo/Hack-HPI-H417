#packages
from torch import manual_seed
from torch import cat as Tcat
import torch.nn as TNN
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean


def build_model_mutag(dataset, report):
    edge_model = EdgeModel_mutag(64, dataset)
    node_model = NodeModel_mutag(64, dataset)
    global_model = GlobalModel_mutag(64, dataset)
    NUM_PASSES = 2

    gn_model = GN_mutag(edge_model, node_model, global_model, dataset, NUM_PASSES)

    if report:
        print('model built:')
        print(gn_model)

    optimizer = Adam(gn_model.parameters(), lr=0.01)
    loss_function = TNN.CrossEntropyLoss()

    return gn_model, optimizer, loss_function


class GN_mutag(TNN.Module):
    def __init__(self, edge_model, node_model, global_model, dataset,
                 num_passes):
        super().__init__()
        manual_seed(12345)
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.lin = TNN.Linear(num_features, dataset.num_classes)
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
                batch[src_node_idx])

            node_attr, edge_out_bar = self.node_model(node_attr, edge_index,
                                                      edge_attr, u, batch)

            u = self.global_model(node_attr, edge_out_bar, u, batch)

        # 2. Readout layer
        graph_attr = Tcat([node_attr, edge_out_bar, u[batch]], dim=1)
        graph_attr_pooled = global_mean_pool(graph_attr, batch)

        return self.lin(graph_attr_pooled)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')


class EdgeModel_mutag(TNN.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        manual_seed(12345)
        num_features = dataset.num_features * 2 + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.edge_mlp = TNN.Sequential(
            TNN.Linear(num_features, hidden_channels), TNN.ReLU(),
            TNN.Dropout(0.5),
            TNN.Linear(hidden_channels, dataset.num_edge_features), TNN.ReLU())

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = Tcat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel_mutag(TNN.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        manual_seed(12345)
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.node_mlp = TNN.Sequential(
            TNN.Linear(num_features, hidden_channels), TNN.ReLU(),
            TNN.Dropout(0.5), TNN.Linear(hidden_channels,
                                         dataset.num_features), TNN.ReLU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes in all graphs of the batch.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        dest_node_idx = edge_index[1]
        # Average all attr of incoming edges for each dest node
        edge_out_bar = scatter_mean(src=edge_attr, index=dest_node_idx, dim=0)
        out = Tcat([x, edge_out_bar, u[batch]], 1)
        return self.node_mlp(out), edge_out_bar


class GlobalModel_mutag(TNN.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        manual_seed(12345)
        num_features = dataset.num_features + dataset.num_edge_features
        num_features += dataset[0].u.size(1)
        self.global_mlp = TNN.Sequential(
            TNN.Linear(num_features, hidden_channels), TNN.ReLU(),
            TNN.Dropout(0.5), TNN.Linear(hidden_channels, 1), TNN.ReLU())

    def forward(self, node_attr_prime, edge_out_bar, u, batch):
        # node_attr_bar: [N, F_x], where N is the number of nodes in the batch.
        # edge_attr: [N, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # Average all node attributes for each graph, using batch tensor.
        node_attr_bar = scatter_mean(node_attr_prime, batch, 0)
        # Average all edge attributes for each graph, using batch
        edge_attr_bar = scatter_mean(edge_out_bar, batch, 0)
        out = Tcat([u, node_attr_bar, edge_attr_bar], dim=1)
        return self.global_mlp(out)