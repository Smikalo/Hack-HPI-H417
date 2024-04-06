from dash import Dash, dcc, html, Output, Input, State
import numpy
# from osmnx.truncate import truncate_graph_bbox
#from osmnx.speed import add_edge_speeds, add_edge_travel_times
from osmnx.distance import nearest_nodes, shortest_path #, k_shortest_paths
import plotly.express as px
# import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
from geopy.geocoders import Nominatim

# Initialize the app
app = Dash(__name__)

def load_graph(fname):
    print("Loading graph") #, end= ' - ')
    with open(fname, "rb") as file_handle:
        G = pickle.load(file_handle)
    #G = add_edge_speeds(G)
    #G = add_edge_travel_times(G)
    return G
# Gfull = load_graph(fname = '/home/onur/WORK/DS/repos/AccIndex.git/data/Berlin_v221217_NodeR0_nhop2/Graphs_splityearly_T1821_V1920_T21/nbhs_graphs_nhop2/train60000val15000test15000/models/predictions/global_edge_preds_G.pkl')
Gfull = load_graph(fname = './data/global_edge_preds_G.pkl')

def get_route(G, C_orig, C_dest, algorithm='osmnx'):
    NodeID_orig = nearest_nodes(G, C_orig.longitude, C_orig.latitude)
    NodeID_dest = nearest_nodes(G, C_dest.longitude, C_dest.latitude)
    if algorithm == 'osmnx':
        route = shortest_path(G, NodeID_orig, NodeID_dest)  #requires sklearn
    return route

app.layout = html.Div(
    children =
        [
        html.Div(
        className='row',  # Define the row element
        children=
            [
            html.Div(className='four columns div-user-controls',
            children = 
                [
                html.H2('AccIndex'),
                html.P('''Enter addresses in Berlin:'''),
                html.Div(
                    dcc.Location(
                    id='url',
                    refresh=False)),
                html.Div(
                    dcc.Input(
                    id='source_location',
                    placeholder='Enter the source address.',
                    type='text',
                    value='Eislebener Str. 4 Berlin'
                    ),
                    html.Br(),
                    dcc.Input(
                    id='destination_location',
                    placeholder='Enter the destination address.',
                    type='text',
                    value='Sonnenburger Str. 73 Berlin'
                )),
                html.Button('Submit',id='button', n_clicks=0, style={}),
                html.P('\n')                             
                ]),  
            # Define the left element
            # Define the right element
            html.Div(id='output_text'),
            html.Div(id='output_graph')
            ])
    ])

# Define the callback for the submit button
@app.callback(
    Output('output_text', 'children'),
    Output('output_graph', 'children'),
    [Input('button', 'n_clicks')],
    [State('source_location', 'value'),
    State('destination_location', 'value')])


def update_output(n_clicks, source_location, destination_location):
    # Code to fetch the shortest path between source and destination location
    #making an instance of Nominatim class
    geolocator = Nominatim(user_agent="my_request")
    source_lat_long = geolocator.geocode(source_location)
    destination_lat_long = geolocator.geocode(destination_location)

    print('source location: %s; coords: %5.2f, %5.2f'%(source_location,source_lat_long.latitude, source_lat_long.longitude))
    print('dest location: %s; coords: %5.2f, %5.2f'%(destination_location,destination_lat_long.latitude, destination_lat_long.longitude))
    
    print('finding shortest path')
    # Retreive the shortest route
    route = get_route(Gfull, source_lat_long, destination_lat_long, 'osmnx')

    accident_count = 0
    lons = []
    lats = []
    description = []
    for node_i in range(len(route)-1):
        node_data = Gfull.nodes[route[node_i]]
        edge_data = Gfull.get_edge_data(route[node_i],route[node_i+1])
        lons.append(node_data['x'])
        lats.append(node_data['y'])
        
        if not 'pred_acc' in edge_data[0]:
            pred_acc = 0
            acc_count_i = 0
        else:
            pred_acc = edge_data[0]['pred_acc']
            if isinstance(pred_acc,numpy.float64):
                #print('print node: %d, pred. acc: %s'%(node_i, str(edge_data[0]['pred_acc'])))
                pred_acc = float(pred_acc)
                acc_count_i  = pred_acc
            else:
                pred_acc = 0.0
                acc_count_i = 0
        description.append('%5.2f acc/y'%pred_acc)
        accident_count += acc_count_i
    displaystr = 'Predicted cumulative accident frequency along the route:%5.1f/year'%accident_count
    # Create a dataframe with the shortest path
    df = pd.DataFrame({
        "lat": lats, #[source_lat_long.latitude, destination_lat_long.latitude],
        "lon": lons, #[source_lat_long.longitude, destination_lat_long.longitude],
        "description": description #["Source", "Destination"]
    })
    #fig = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name="description", color_discrete_sequence=["fuchsia"], zoom=10, height=300)
    fig = px.line_mapbox(df, lat="lat", lon="lon", hover_name="description", color_discrete_sequence=["blue"], zoom=11, height=500) #, width = 900
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    render_option = 'interactive'
    if render_option == 'static':
        from base64 import b64encode
        img_bytes = fig.to_image(format="png")
        encoding = b64encode(img_bytes).decode()
        img_b64 = "data:image/png;base64," + encoding
        return displaystr,html.Img(src=img_b64, style={'height': '500px'})
    elif render_option == 'interactive':
        return displaystr,dcc.Graph(figure=fig)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)