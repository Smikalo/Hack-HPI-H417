#import packages
import os
import streamlit as st

# import local libraries
from visualizations import *
from user_input import *
from graph_funcs import *


def main(parameters, StLit=True):

    #Load graph object (only if not already loaded, as it takes times)
    if StLit:
        if "loaded" not in st.session_state:
            st.session_state.loaded = True
            st.write('Preparing the environment..')
        Gfull = load_graph(parameters)
    else:
        print('Loading graph')
        Gfull = load_graph(parameters)
        go = True #False

    # user input
    C_orig, C_dest, go = get_user_input(StLit)  # C_ is for coordinates
    
    if go:
        print('got addresses', end=' - ')

        #Truncate the graph based on coordinates
        G = truncate_graph(Gfull, C_orig, C_dest)

        # Retreive the shortest route, if required
        routes = get_routes(G, C_orig, C_dest, parameters['route_alg'])

        # Plot map, show the coordinates of the chosen points
        plot_map_box(G, routes, C_orig, C_dest, parameters)

        if StLit:
            st.image('plot_map_route.png')

        print('done')

#@st.cache(suppress_st_warning=True, allow_output_mutation=False)
@st.experimental_singleton(suppress_st_warning=True)
def load_graph(parameters):
    print("Loading graph", end= ' - ')
    with open(parameters['G_fname'], "rb") as file_handle:
        G = pickle.load(file_handle)
    G = add_edge_speeds(G)
    G = add_edge_travel_times(G)
    return G

if __name__ == "__main__":
    parameters = {
        # 'G_fname': os.path.join('data',
        #                        'Berlin_with_acc_on_nodes_and_edges.pkl'),
        'G_fname': os.path.join('data',
                                'global_predictions_restored_G.pkl'),
        #'G_fname': None, #if an fname of a pickled graph is not provided, it will be created
        'route_alg': 'osmnx'  #osmnx, osmnx_k, taxicab
    }

    main(parameters)
