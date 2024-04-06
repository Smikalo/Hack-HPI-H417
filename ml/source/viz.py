import networkx as nx
import osmnx as ox
# from torch_geometric.utils.convert import from_networkx
import plotly.graph_objects as go # pip install plotly==5.11.0 #pip install -U kaleido
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
print(os.getcwd())
from source import neighbourhood as nbh



def main_viz():
    print("Starting....................")
    
    datasetrootname = "v221215"
    split_strategy = 'historical' #options: randomsamp, yearly. # TODO: historical
    splitsuf_yearly = 'T1819_V20_T21'
    splitsuf_random = 'T07_V015_T015'
    splitsuf_historical = 'T1819_V1920_T2021_r0'

    dataset_name = "Berlin_" + datasetrootname
    # rootpath = os.path.dirname(os.path.realpath(__file__)) #Abs path of AccIndex.git
    rootpath = os.getcwd()
    data_path = os.path.join(rootpath,'data')
    # splitsuf  
    if split_strategy == 'randomsamp':
        splitsuf = splitsuf_random
    elif split_strategy == 'yearly':
        splitsuf = splitsuf_yearly
    elif split_strategy == 'historical':
        splitsuf = splitsuf_historical
    graphdatapath = os.path.join(data_path, dataset_name, "Graphs_split"+split_strategy+'_'+splitsuf)

    rootfname = os.path.join(graphdatapath,'test' + '.pkl')
    G = pickle.load(open(rootfname, 'rb'))
    # node = random.choice(list(G.nodes))
    # edge = random.choice(list(G.out_edges(node))+list(G.in_edges(node)))
    # lst, node, edge = nbh.nest_of_neighbourhoods_aroud_node_edge(G, node, edge, hop=3, num_nbh=10)
    list_nbh = nbh.neighbourhood_node_edge_tuple(G, n=3, num_nbh=1)
    plot_neighbourhood_node_edge(list_nbh[0][0], list_nbh[0][1], list_nbh[0][2], show=True)#, rootfolder, rootfname)
    # plot_nbh_on_osmnx_map(G, list_nbh[0][0], list_nbh[0][1], list_nbh[0][2], show=True)#, rootfolder, rootfname)
    plot_nbh_node_edge_on_map(G,  N=list_nbh[0][0], node=list_nbh[0][1], edge=list_nbh[0][2], show=True)#, rootfolder=rootfolder, rootfname=rootfname)
    print('done.')
    return


def plot_neighbourhood_node_edge(N, node=None, edge=None, show=True, figures_folder_path=None):
    '''
    Plots the neighbourhood graph with the node and edge highlighted.

    :param N: neighbourhood graph
    :param node: node id
    :param edge: edge tuple
    :param rootfolder: root folder to save the figure
    :param rootfname: root file name to save the figure
    :param return: plot of the neighbourhood graph with the node and edge highlighted
    '''
    pos = nx.spring_layout(N)
    nx.draw(N, pos, with_labels=True)
    if node:
        nx.draw_networkx_nodes(N, pos, nodelist=[node], node_color='r')
    if edge:
        nx.draw_networkx_edges(N, pos, edgelist=[edge], edge_color='r')
    if show == True:
        plt.show()
    if figures_folder_path:
        figfname = os.path.join(figures_folder_path+f'_nbh_nx_graph_{node}.png')
        plt.savefig(figfname)
        print(f'figure saved: {figfname}')


def plot_nbh_node_edge_on_map(G, N, node, edge, show=True, figures_folder_path=None):
    '''
    Plots the neighbourhood graph with the node and edge highlighted on the open-street-map.

    G: networkx graph
    N: neighbourhood graph
    node: node id
    edge: edge tuple
    show: show the plot
    rootfolder: root folder to save the figure
    rootfname: root file name to save the figure
    ----------------
    return: plot of the neighbourhood graph with the node and edge highlighted on the map of the original graph
    '''
    # N_undi = N.to_undirected()
    # for i, d in enumerate(N_undi.edges()):
    for i, d in enumerate(N.edges()):
        # print(d)
        u = d[0]
        v = d[1]
        if i == 0:
            edge_lon = [] 
            edge_lat = []  
            
            edge_lon = [G.nodes[u]['x'], G.nodes[v]['x']]
            edge_lat = [G.nodes[u]['y'], G.nodes[v]['y']]
            
            data = go.Scattermapbox(
                name = "path",
                mode = "lines+text",
                lon = edge_lon,
                lat = edge_lat,
                marker = dict(size=20, color='green'),
                line = dict(width = 4.5, color = 'blue'),
                textposition='top right',
                textfont=dict(size=16, color='black'),
                text= [d])
            fig = go.Figure(data=data)
        else:
            edge_lon = [] 
            edge_lat = []  
                        
            edge_lon = [G.nodes[u]['x'], G.nodes[v]['x']]
            edge_lat = [G.nodes[u]['y'], G.nodes[v]['y']]
        
            fig.add_trace(go.Scattermapbox(
                name = "path",
                mode = "lines+text",
                lon = edge_lon,
                lat = edge_lat,
                # marker = {'size': 10},
                # line = dict(width = 4.5, color = 'blue'),
                marker = dict(size=20, color='green'),
                line = dict(width = 4.5, color = 'blue'),
                textposition='top right',
                textfont=dict(size=16, color='black'),
                text= [d]))

    node_lon = [G.nodes[node]['x']]
    node_lat = [G.nodes[node]['y']]
    fig.add_trace(go.Scattermapbox(
        name = "Source",
        mode = "markers",
        lon = node_lon,
        lat = node_lat,
        marker = {'size': 20, 'color':"yellow"},
        textposition='top right',
        textfont=dict(size=18, color='Red'),
        text= [node]))

    u = edge[0]
    v = edge[1]
    edge_lon = [G.nodes[u]['x'], G.nodes[v]['x']]
    edge_lat = [G.nodes[u]['y'], G.nodes[v]['y']]
    fig.add_trace(go.Scattermapbox(
        name = "path",
        mode = "lines",
        lon = edge_lon,
        lat = edge_lat,
        marker = {'size': 10},
        line = dict(width = 5, color = 'red')))

    long = [] 
    lat = []  
    # for i in N_undi.nodes():
    for i in N.nodes():
        point = G.nodes[i]
        long.append(point['x'])
        lat.append(point['y'])

    fig.add_trace(go.Scattermapbox(
        name = "Point",
        mode = "markers",
        lon = long,
        lat = lat,
        # marker = {'size': 10, 'color':"green"}))
        marker = dict(size=12, color='green'),
        line = dict(width = 4.5, color = 'blue'),
        textposition='top right',
        textfont=dict(size=16, color='black'),
        text= [i for i in N.nodes()]))


    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)

    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="open-street-map",#"stamen-terrain",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    # adjsting the center of the plot, removing the margin and legend
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox = {
                        'center': {'lat': lat_center, 
                        'lon': long_center},
                        'zoom': 20},
                    showlegend=False)
    if show == True:
        fig.show()
    if figures_folder_path:
        figfname = os.path.join(figures_folder_path+f'_nbh_map_{node}.png')
        fig.write_image(figfname) #pip install -U kaleido
        print(f'figure saved: {figfname}')


def plot_NEST_nbh_node_edge_on_map(G, N_lst, node, edge, show=True):#, rootfolder=None, rootfname=None):
    '''
    G: networkx graph
    N: list neighbourhood graph around node and edge
    node: node id
    edge: edge tuple
    show: show the plot
    rootfolder: root folder to save the figure
    rootfname: root file name to save the figure
    ----------------
    return: plot of the neighbourhood graph with the node and edge highlighted on the map of the original graph
    '''
    fig = go.Figure()
    # node_lon = 
    # node_lat = 
    fig.add_trace(go.Scattermapbox(
        name = "Source",
        mode = "markers",
        lon = [G.nodes[node]['x']],
        lat = [G.nodes[node]['y']],
        marker = {'size': 20, 'color':"red"},
        textposition='top right',
        textfont=dict(size=18, color='Red'),
        text= [node]))
    
    for N in N_lst:
        for d in N.edges():
            # print(d)
            u = d[0]
            v = d[1]

            edge_lon = [] 
            edge_lat = []  
                        
            edge_lon = [G.nodes[u]['x'], G.nodes[v]['x']]
            edge_lat = [G.nodes[u]['y'], G.nodes[v]['y']]
        
            fig.add_trace(go.Scattermapbox(
                name = "path",
                mode = "lines+text",
                lon = edge_lon,
                lat = edge_lat,
                # marker = {'size': 10},
                # line = dict(width = 4.5, color = 'blue'),
                marker = dict(size=20, color='green'),
                line = dict(width = 4.5, color = 'blue'),
                textposition='top right',
                textfont=dict(size=16, color='black'),
                text= [d]))
        N_long = [] 
        N_lat = []
        for i in N.nodes():
            point = G.nodes[i]
            N_long.append(point['x'])
            N_lat.append(point['y'])

        fig.add_trace(go.Scattermapbox(
            name = "Point",
            mode = "markers",
            lon = N_long,
            lat = N_lat,
            # marker = {'size': 10, 'color':"green"}))
            marker = dict(size=12, color='green'),
            line = dict(width = 4.5, color = 'blue'),
            textposition='top right',
            textfont=dict(size=16, color='black'),
            text= [i for i in N.nodes()]))

    u = edge[0]
    v = edge[1]
    edge_lon = [G.nodes[u]['x'], G.nodes[v]['x']]
    edge_lat = [G.nodes[u]['y'], G.nodes[v]['y']]
    fig.add_trace(go.Scattermapbox(
        name = "path",
        mode = "lines",
        lon = edge_lon,
        lat = edge_lat,
        marker = {'size': 10},
        line = dict(width = 5, color = 'red')))

    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="open-street-map",#"stamen-terrain",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    # adjusting the center of the plot, removing the margin and legend
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox = {
                        'center': {'lat': G.nodes[node]['y'], 
                        'lon': G.nodes[node]['x']},
                        'zoom': 15},
                    showlegend=False)
    if show == True:
        fig.show()

#put the findings in slides.. create few examples for understanding.. Seed, Node, Edge

def plot_nbh_on_osmnx_map(G, N, node, edge, show=True, figures_folder_path=None):
    '''
    Plot the neighbourhood graph on the osmnx map with the node and edge highlighted
    
    :param G: osmnx graph
    :param N: neighbourhood graph
    :param node: node id
    :param edge: edge tuple
    :param rootfolder: root folder to save the figure
    :param rootfname: root file name to save the figure
    :param return: plot of the neighbourhood graph on the osmnx map with the node and edge highlighted
    '''
    ec = ['red' if key == edge else 'orange' if key in N.edges() else 'gray' for key in G.edges()] 
    nc = ['red' if key == node else 'green' if key in N.nodes() else 'white' for key in G.nodes()]
    fig, ax = ox.plot_graph(G, node_color=nc , node_edgecolor='k', node_size=5, 
                        node_zorder=3, edge_color=ec, edge_linewidth=2)
    if show == True:
        fig.show()
    if figures_folder_path:
        figfname = os.path.join(figures_folder_path+f'_nbh_osmnx_map_{node}.png')
        plt.savefig(figfname)
        print(f'figure saved: {figfname}')


def plot_path(G, path, acc_dict_nodes, acc_dict_edges):
    '''
    Plot the path on the osmnx map with the node and edge highlighted

    :param G: osmnx graph
    :param path: path
    :param acc_dict_nodes: accumulated node dictionary
    :param acc_dict_edges: accumulated edge dictionary
    :param return: plot of the path on the osmnx map with the node and edge highlighted
    '''
    
    long = [] 
    lat = []  
    for i in path:
        point = G.nodes[i]
        long.append(point['x'])
        lat.append(point['y'])
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    layout = {
        "mapbox_style":"open-street-map",#"stamen-terrain", 'carto-positron', 'carto-darkmatter', 'stamen-toner', 'stamen-watercolor'
        "margin":{"r":0,"t":0,"l":0,"b":0},
        "mapbox":{'center': {'lat': lat_center, 'lon': long_center},'zoom': 15},
        # "showlegend": False,
        "hovermode":'closest'
    }
    # adding path
    trace1={
        "type":"scattermapbox",
        "name": "Route",
        "mode": "lines",
        "lon": long,
        "lat": lat,
        "line": dict(width = 4.5, color='blue')
    }

    origin_point = G.nodes[path[0]]
    destination_point = G.nodes[path[-1]]
    # adding source marker
    trace2={
        "type": "scattermapbox",
        "name": "Source",
        "text": "Source",
        "mode": "markers",
        "lon": [origin_point['x']],
        "lat": [origin_point['y']],
        "marker": {'size': 20, 'color':"red"}
    }
    # adding destination marker
    trace4={
        "type": "scattermapbox",
        "name": "Destination",
        "text": "Destination",
        "mode": "markers",
        "lon": [destination_point['x']],
        "lat": [destination_point['y']],
        "marker": {'size': 20, 'color':'black'}
    }

    text = [f"Node: {j}, #Acc: {acc_dict_nodes[j]}<br> Edge: {(j,path[i+1])}, # Acc: {acc_dict_edges[(j,path[i+1])]}"  for i,j in enumerate(path[:-1])]
    trace3={
        "type": "scattermapbox",
        "name": "Nodes",
        "mode": "markers",
        "lon": long,
        "lat": lat,
        "marker": {'showscale':False,
                    'size': 10, #list(map(lambda x: x * 40, list(acc_dict_nodes.values()))),
                    'color': 'green'},
        "textposition":'top right',
        "textfont":dict(size=16, color='black'),
        "text": text 
    }


    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(hoverinfo='text', selector=dict(type='scattermapbox'))

    fig.show()





# if __name__=='__main__':
#     main_viz()
