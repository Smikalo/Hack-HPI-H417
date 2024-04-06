import networkx as nx
import osmnx as ox
import random
from torch_geometric.utils.convert import from_networkx


def random_points_on_map_deprecated(G, num_random_points):
    """Generate random points within the graph's bounding box. Then tuples (lat, lon) are generated and nearest nodes in the graph G are found.
    Parameters
    ----------
    G : networkx multidigraph
    num_random_points : int
        number of random points to generate
    Returns
    -------
    list : list of nodes from the graph G
    """
    points = ox.utils_geo.sample_points(
        ox.get_undirected(G), num_random_points).to_crs("EPSG:4326")
    list_random_points = list()
    for i in range(num_random_points):
        list_random_points = list_random_points + list(points.iloc[i].coords)
    return [ox.nearest_nodes(G, x[0], x[1]) for x in list_random_points]


def n_neighbourhood_deprecated(G, node, hop):
    '''
    G: networkx graph
    node: node id
    hop: number of hops
    ----------------
    return: subgraph (nx.subgraph) of G with n-hop neighborhood of node
    '''
    neighbourhood = nx.single_source_dijkstra_path_length(G, node, cutoff=hop)
    # nbh_subgraph = G.subgraph(node_list)
    node_list = [node for node, length in neighbourhood.items()]
    # copy() is necessary to avoid changing the original graph and its attributes. Its an unfreezed graph.
    nbh_subgraph = G.subgraph(node_list).copy()
    return nbh_subgraph


def random_node_edge_deprecated(G):
    '''
    G: networkx graph
    ----------------
    return: random node and edge from G
    '''
    node = random.choice(list(G.nodes))
    edge = random.choice(list(G.out_edges(node)))
    return node, edge
