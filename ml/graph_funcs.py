# import packages
from osmnx import graph_from_bbox
from osmnx.truncate import truncate_graph_bbox
from osmnx.speed import add_edge_speeds, add_edge_travel_times
from osmnx.distance import nearest_nodes, shortest_path, k_shortest_paths
#import taxicab as tc
import pickle
import streamlit as st


def truncate_graph(G, c1, c2):
    #st.write(" Truncating graph based on user coordinates")
    pad = 0.005
    truncate_graph_bbox(G,
                        max([c1.latitude, c2.latitude]) + pad,
                        min([c1.latitude, c2.latitude]) - pad,
                        max([c1.longitude, c2.longitude]) + pad,
                        min([c1.longitude, c2.longitude]) - pad)
    return G


def get_routes(G, C_orig, C_dest, algorithm='osmnx'):
    NodeID_orig = nearest_nodes(G, C_orig.longitude, C_orig.latitude)
    NodeID_dest = nearest_nodes(G, C_dest.longitude, C_dest.latitude)
    if algorithm == 'osmnx':
        route = shortest_path(G, NodeID_orig, NodeID_dest)  #requires sklearn
        routes = [route]
    elif algorithm == 'osmnx_k':
        routes_go = k_shortest_paths(
            G,
            NodeID_orig,
            NodeID_dest,
            k=3  #default
        )
        routes = list(routes_go)  #convert the generator object to list
        st.write(f' shortest {len(routes)} routes are computed')
    elif algorithm == 'taxicab':
        # taxicab solution
        #from : https://stackoverflow.com/questions/62637344/osmnx-is-there-a-way-to-find-an-accurate-shortest-path-between-2-coordinates
        route = tc.distance.shortest_path(G, C_orig, C_dest)
        #problem: sometimes it cannot find any route
        routes = [route]
    else:
        raise (ValueError(f'unknown route finding algorithm: {algorithm}'))
    return routes
