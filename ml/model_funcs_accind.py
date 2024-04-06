# packages
from torch import manual_seed
from torch import cat as Tcat
import torch.nn as TNN
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean


def build_model_accind(DSpars, NUM_PASSES=2, report=False):
    edge_model = EdgeModel_accind(64, DSpars)
    node_model = NodeModel_accind(64, DSpars)
    global_model = GlobalModel_accind(64, DSpars)

    gn_model = GN_accind(edge_model, node_model,
                         global_model, DSpars, NUM_PASSES)

    if report:
        print('model built:')
        print(gn_model)

    optimizer = Adam(gn_model.parameters(), lr=0.005)
    loss_function = TNN.MSELoss()
    test_loss_function = TNN.L1Loss()
    # may already work, if not, own implementation as in:
    # https://donaldpinckney.com/books/pytorch/book/ch2-linreg/2018-03-21-multi-variable.html

    return gn_model, optimizer, loss_function, test_loss_function


class GN_accind(TNN.Module):
    def __init__(self, edge_model, node_model, global_model, DSpars,
                 num_passes):
        super().__init__()
        manual_seed(12345)
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        num_features = DSpars.num_features + DSpars.num_edge_features
        num_features += DSpars.u_size_1
        self.lin = TNN.Linear(num_features, DSpars.num_y)
        self.num_passes = num_passes
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, node_attr, edge_attr, u, edge_index, n_pred_idx, e_pred_idx, batch):
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
        # 2 Predict for node and edge
        # for the mutag case, u needs to be broadcasted (u[batch]) to the number of total nodes and edges
        #node_edge_attr = Tcat([node_attr[n_pred_idx], edge_attr[e_pred_idx], u[batch]], dim=1)
        # but in our case we don't need to broadcast the u, as for each graph, we have only one node and one edge
        node_edge_attr = Tcat(
            [node_attr[n_pred_idx], edge_attr[e_pred_idx], u], dim=1)
        # output must be dim 1 x 2 and input should be num_node_feat + num_edge_feat
        out = self.lin(node_edge_attr)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')


class EdgeModel_accind(TNN.Module):
    def __init__(self, hidden_channels, DSpars):
        super().__init__()
        manual_seed(12345)
        num_features = DSpars.num_features * 2 + DSpars.num_edge_features
        num_features += DSpars.u_size_1
        self.edge_mlp = TNN.Sequential(
            TNN.Linear(num_features, hidden_channels), TNN.ReLU(),
            TNN.Dropout(0.25),
            TNN.Linear(hidden_channels, DSpars.num_edge_features), TNN.ReLU())

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = Tcat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel_accind(TNN.Module):
    def __init__(self, hidden_channels, DSpars):
        super().__init__()
        manual_seed(12345)
        num_features = DSpars.num_features + DSpars.num_edge_features
        num_features += DSpars.u_size_1
        self.node_mlp = TNN.Sequential(
            TNN.Linear(num_features, hidden_channels), TNN.ReLU(),
            TNN.Dropout(0.25), 
            TNN.Linear(hidden_channels, DSpars.num_features), TNN.ReLU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes in all graphs of the batch.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        dest_node_idx = edge_index[1]
        # Average all attr of incoming edges for each dest node
        edge_out_bar = scatter_mean(src=edge_attr, index=dest_node_idx,  out = edge_attr.new_zeros((x.shape[0], edge_attr.shape[1])) ,dim=0)
        out = Tcat([x, edge_out_bar, u[batch]], 1)
        return self.node_mlp(out), edge_out_bar


class GlobalModel_accind(TNN.Module):
    def __init__(self, hidden_channels, DSpars):
        super().__init__()
        manual_seed(12345)
        num_features = DSpars.num_features + DSpars.num_edge_features
        num_features += DSpars.u_size_1
        self.global_mlp = TNN.Sequential(
            TNN.Linear(num_features, hidden_channels), TNN.ReLU(),
            TNN.Dropout(0.25), 
            TNN.Linear(hidden_channels, DSpars.u_size_1), TNN.ReLU())

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
