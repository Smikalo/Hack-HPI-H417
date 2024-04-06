import networkx as nx
import osmnx as ox
import random
from torch_geometric.utils.convert import from_networkx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPoint
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def n_neighbourhood(G, node, hop):
    '''
    Cretes a subgraph of G with n-hop neighborhood of node
    
    :param G: networkx graph
    :param node: node id
    :param hop: number of hops
    :param return: subgraph (nx.subgraph) of G with n-hop neighborhood of node
    '''
    hop_nodes = [node]
    i=0
    while i < hop :
        node_list = [n for node in hop_nodes for n in nx.all_neighbors(G, node)]
        hop_nodes = hop_nodes + node_list
        i+=1
    hop_nodes = set(hop_nodes) #keeps only unique nodes
    nbh_subgraph = G.subgraph(hop_nodes).copy() # copy() is necessary to avoid changing the original graph and its attributes. Its an unfreezed graph.
    nbh_subgraph.node_attributes_list = G.node_attributes_list
    nbh_subgraph.edge_attributes_list = G.edge_attributes_list
    return nbh_subgraph


def neighbourhood_creation(G, n, num_nbh):
    '''
    Creates a list of num_nbh number of n-hop neighbourhoods in G

    :param G: networkx graph
    :param n: number of hops
    :param num_nbh: number of neighbourhoods to generate
    :param return: list of subgraphs (nx.subgraph) of G with n-hop neighbourhood of node
    '''
    random_points_on_map = random.choices([n for n in G.nodes()], k=num_nbh)
    list_nbh = list()
    for node in random_points_on_map:
        list_nbh.append(n_neighbourhood(G, node, n))
    return list_nbh


def random_node_edge(G):
    '''
    Returns a random node and edge from G

    :param G: networkx graph
    :param return: random node and edge from G
    '''
    node = random.choice(list(G.nodes))
    edge = random.choice(list(G.out_edges(node))+list(G.in_edges(node)))
    return node, edge


def neighbourhood_node_edge_tuple(G, n, num_nbh):
    '''
    Creates a list of num_nbh number of of tuples (nbh, node, edge), of n-hops neighbourhood from G, a random node and edge from the respective neighbourhood
    :param G: networkx graph
    :param n: number of hops
    :param num_nbh: number of neighbourhoods to generate
    :param return: list of tuples (nbh, node, edge) from the subgraphs (nx.subgraph) of G with n-hop neighbourhood of node
    '''
    list_node_edge = list()
    list_nbh = neighbourhood_creation(G, n, num_nbh)
    for neighbourhood in list_nbh:
        node, edge = random_node_edge(neighbourhood)
        list_node_edge.append((neighbourhood, node, edge))
    return list_node_edge


def datapoints_neighbourhood_with_labels(G, density_df, n, num_nbh, split=None):
    '''

    G: networkx graph
    n: number of hops
    num_nbh: number of neighbourhoods to generate
    ----------------
    return: list of tuples (nbh, node, edge) from the subgraphs (nx.subgraph) of G with n-hop neighborhood of node
    '''
    list_nbh_node_edge = neighbourhood_node_edge_tuple(G, n, num_nbh)
    totnum_nbh = len(list_nbh_node_edge)
    datapoints = [None] * totnum_nbh
    portionfinishedth=0
    print(f' % of total ({totnum_nbh}) neighborhoods created:')
    if split is not None:
        for ti,t in enumerate(list_nbh_node_edge):
            datapoints[ti] = attr_aggregation_cleaning_converting_N(t, density_df, split)
            #print progress:
            if (100*ti/totnum_nbh)>portionfinishedth:
                print (f'  {portionfinishedth}%')
                portionfinishedth += 10

    else:    
        for ti,t in enumerate(list_nbh_node_edge):
            datapoints[ti] = attr_aggregation_cleaning_converting_N(t, density_df)
            #print progress:
            if (100*ti/totnum_nbh)>portionfinishedth:
                print (f'  {portionfinishedth}%')
                portionfinishedth += 10

    

    return datapoints

def scaling_collect(datapoints):
    '''
    Collects datapoints for fitting and/or transforming the scaler

    :param datapoints: list of tuples (nbh, node, edge) 
    :param return: list of variables to be fitted and/or transformed
    '''
    # Collects datapoints for fitting and/or transforming the scaler
    density = []
    coords = []
    for n in datapoints:
        density.append([n.u[0][0]])         # the precise assignment heavily depends on the initial order of the variables
        coords.append(n.u[0][1:3].tolist()) # in the global attribute vector determined in add_global_attr()
    density = np.array(density)
    coords = np.array(coords)

    return density, coords

def scaling_fit(density, coords):
    '''
    Fits the scalers to the variables collected in scalling_collect()
    
    :param density: list of density values
    :param coords: list of coordinates
    :param return: fitted scalers
    '''
    # fits the scalers to the variables collected in scalling_collect()
    density_minmax_scaler = MinMaxScaler()
    coords_standard_scaler = StandardScaler()
    density_minmax_scaler.fit(density)
    coords_standard_scaler.fit(coords)

    return density_minmax_scaler, coords_standard_scaler

def scaling_transform(datapoints, density, coords, density_minmax_scaler, coords_standard_scaler):
    '''
    Transforms the variables collected in scaling_collect() by applying the scalers from scaling_fit() and assigns
    the transformed values back to the original graph objects

    :param datapoints: list of tuples (nbh, node, edge)
    :param density: list of density values
    :param coords: list of coordinates
    :param density_minmax_scaler: fitted minmax scaler
    :param coords_standard_scaler: fitted standard scaler
    :param return: transformed datapoints
    '''
    # Transforms the variables collected in scaling_collect() by applying the scalers from scaling_fit() and assigns
    # the transformed values back to the original graph objects
    scaled_density = density_minmax_scaler.transform(density)
    scaled_coords = coords_standard_scaler.transform(coords)
    for i, n in enumerate(datapoints):
        n.u[0][0] = float(scaled_density[i][0])
        n.u[0][1:3] =  torch.from_numpy(scaled_coords[i])

    return datapoints


def attr_aggregation_cleaning_converting_N(t, density_df, split=None):
    '''
    Aggregates the node and edge attributes, cleans the graph, and converts it to a torch_geometric graph.

    :param t: tuple (nbh, node, edge)
    :param return: torch_geometric graph with the aggregated node and edge attributes
    '''
    N = t[0]
    n_pred = t[1]
    e_pred = t[2]
    N.graph.clear()  # clearing the graph attributes
    n_pred_idx, e_pred_idx, label_tuple = get_indices_label(N, n_pred, e_pred, split)
    # adding the labels to the graph in list form
    N.graph.update(y=[[label_tuple[0], label_tuple[1]]])
    # adding the node index to the graph
    N.graph.update(n_pred_idx=n_pred_idx)
    # adding the edge index to the graph
    N.graph.update(e_pred_idx=e_pred_idx)
    N = add_global_attr(N, density_df, n_pred)  # adding the global attributes to the graph
    # aggregating the node and edge attributes to the graph
    attr_aggregation_cleaning_N(N)
    # converting the networkx graph to torch_geometric graph
    H = from_networkx(N)

    return H


def get_indices_label(N, n_pred, e_pred, split=None):
    '''
    Gets the indices of the node and edge for the prediction task and the labels for the prediction task.
    ----------------
    :param N: neighbourhood graph
    :param n_pred: node id
    :param e_pred: edge tuple
    :param return: index of the node and edge for the prediction task, label tuple (lable for node, label for edge) 
    '''
    if split == 'train':
        n_pred_idx = None
        label_tuple = tuple()
        for i, (n, d) in enumerate(N.nodes(data=True)):
            if n == n_pred:
                n_pred_idx = i
                label_tuple += (d['nr_accidents_19'],)
                break
        e_pred_idx = None
        for i, (u, v, d) in enumerate(N.edges(data=True)):
            if (u, v) == e_pred:
                e_pred_idx = i
                label_tuple += (d['nr_accidents_19'],)
                break

    elif split == 'val':
        n_pred_idx = None
        label_tuple = tuple()
        for i, (n, d) in enumerate(N.nodes(data=True)):
            if n == n_pred:
                n_pred_idx = i
                label_tuple += (d['nr_accidents_20'],)
                break
        e_pred_idx = None
        for i, (u, v, d) in enumerate(N.edges(data=True)):
            if (u, v) == e_pred:
                e_pred_idx = i
                label_tuple += (d['nr_accidents_20'],)
                break
    elif split == 'test':
        n_pred_idx = None
        label_tuple = tuple()
        for i, (n, d) in enumerate(N.nodes(data=True)):
            if n == n_pred:
                n_pred_idx = i
                label_tuple += (d['nr_accidents_21'],)
                break
        e_pred_idx = None
        for i, (u, v, d) in enumerate(N.edges(data=True)):
            if (u, v) == e_pred:
                e_pred_idx = i
                label_tuple += (d['nr_accidents_21'],)
                break
    else:
        n_pred_idx = None
        label_tuple = tuple()
        for i, (n, d) in enumerate(N.nodes(data=True)):
            if n == n_pred:
                n_pred_idx = i
                label_tuple += (d['nr_accidents'],)
                break
        e_pred_idx = None
        for i, (u, v, d) in enumerate(N.edges(data=True)):
            if (u, v) == e_pred:
                e_pred_idx = i
                label_tuple += (d['nr_accidents'],)
                break

    return n_pred_idx, e_pred_idx, label_tuple


def attr_aggregation_cleaning_N(N):
    '''
    Aggregates the node and edge attributes and cleans the graph.

    :param N: neighbourhood graph
    :param return: neighbourhood graph with the aggregated node and edge attributes
    '''
    for u, n in N.nodes(data=True):
        n['x'] = tuple([n[a] for a in N.node_attributes_list])
        for k in list(n):
            if k not in ['x']:
                n.pop(k)

    for u, v, e in N.edges(data=True):
        e['edge_attr'] = tuple([e[a] for a in N.edge_attributes_list])
        for k in list(e):
            if k not in ['edge_attr']:
                e.pop(k)

    return N


def add_global_attr(N, density_df, n_pred):
    '''
    Adds the global attributes to the graph.
    
    :param N: neighbourhood graph
    :param density_df: dataframe with the population density and area utilisation of the blocks
    :param n_pred: node id
    :param return: neighbourhood graph with the poulation density, lat-longs of the n_pred and area utilisation as global attributes
    '''
    # Identify the neighbourhood coordinates (the coordinates of the predicted Node)
    nodes_of_N = []
    for i,n in N.nodes(data=True):
        nodes_of_N.append([n['x'], n['y']])
        if i == n_pred:
            nbh_lat = n["x"]
            nbh_lon = n["y"]

    # Find overlap of neighbourhood and ISU5 blocks and compute population density
    nodes_of_N = MultiPoint(nodes_of_N)
    nbh_area = nodes_of_N.convex_hull
    #density_df["intersects"] = density_df['geometry'].apply(lambda x: nbh_area.intersects(x))
    #if density_df[density_df["intersects"]]["flalle"].sum() == 0:
    print("  - zero error: density imputed with average density of Berlin")
    pop_density = 11232 # Average density of Berlin
    #else:
     #   pop_density = 1000**2*density_df[density_df["intersects"]]["ew2021"].sum()/density_df[density_df["intersects"]]["flalle"].sum()

    # Encode area utilisation that is present in neighourhood
    nbh_area_utilisation_vector =[0]#len(density_df['typklar'].value_counts())*[0.]
    #for utilisation in density_df[density_df["intersects"]]["typklar"].unique():
     #   nbh_area_utilisation_vector[list(density_df['typklar'].value_counts().index).index(utilisation)] = 1.

    global_attribute = [float(pop_density), float(nbh_lat), float(nbh_lon) ] + nbh_area_utilisation_vector

    N.graph.update(u=[global_attribute])

    return N



def nest_of_neighbourhoods_aroud_node_edge(G, node, edge, hop, num_nbh):
    '''
    Generates a list of desired number of n-hop neighbourhoods containing node and edge, the given node and edge from the subgraphs of G.

    :param G: networkx graph
    :param node: node id
    :param edge: edge tuple
    :param hop: number of hops
    :param num_nbh: number of neighbourhoods to generate
    :param return: list of desired number of n-hop neighbourhoods containing node and edge, the given node and edge from the subgraphs of G
    '''
    hop_nodes = [node]
    i=0
    while i < hop-1 : 
        node_list = [n for node in hop_nodes for n in nx.all_neighbors(G, node)]
        hop_nodes = hop_nodes + node_list
        i+=1
    hop_nodes = list(set(hop_nodes))
    num_nbh = min(num_nbh, len(hop_nodes))
    # print('all nodes', hop_nodes)
    random_nodes = np.random.choice(np.array(hop_nodes), size=num_nbh, replace=False) #random.choices(hop_nodes, k=num_nbh)
    # print('random nodes', random_nodes)
    nest_of_nbh = [n_neighbourhood(G, n, hop) for n in random_nodes]

    return nest_of_nbh, node, edge

def shortest_path_on_map(G, orig, dest, weight="lenght"): # weight1="length", weight2="travel_time", weight3="nr_accidents"):
    '''
    Returns shortest path from origin to destination w.r.t. weight.

    :param G: networkx graph
    :param orig: origin node id
    :param dest: destination node id
    :param weight: default="length" or "travel_time", "nr_accidents" or "all"
    :param return: shortest paths from origin to destination w.r.t. weight, if weight="all" then returns list of shortest paths w.r.t. lenght, travel_time and nr_accidents respectively.  
    '''
    origin_node = ox.distance.nearest_nodes(G, orig[1], orig[0])
    destination_node = ox.distance.nearest_nodes(G, dest[1], dest[0])
    # shortest path w.r.t. weight1
    if weight=="length":
        path = ox.shortest_path(G, origin_node, destination_node, weight, )
        
    elif weight=="travel_time":
        path = ox.shortest_path(G, origin_node, destination_node, weight)
        
    elif weight=="nr_accidents":
        path = ox.shortest_path(G, origin_node, destination_node, weight)
        
    else:
        path1 = ox.shortest_path(G, origin_node, destination_node, "length")
        path2 = ox.shortest_path(G, origin_node, destination_node, "travel_time")
        path3 = ox.shortest_path(G, origin_node, destination_node, "nr_accidents")
        path = [path1, path2, path3]
    
    return path


def get_node_edge_pairs_from_path(path):
    '''
    Returns list of node and edge tuples from the path.

    :param G: networkx graph
    :param path: shortest path from origin to destination
    :param return: list of node and edge tuples from the path
    '''
    node_edge_pairs = []
    for i in range(len(path)-1):
        node = path[i]
        edge = (path[i], path[i+1])
        node_edge_pairs.append((node, edge))
    
    # node_edge_pairs.append((path[-1], (path[-1], path[-2])))
        
    return node_edge_pairs