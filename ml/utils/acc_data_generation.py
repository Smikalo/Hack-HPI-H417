# library
import pandas as pd
import networkx as nx
import os
import osmnx as ox
import pickle
from pathlib import Path
from data_encoders import encode_categorical_variables
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
# Download the data from  https://unfallatlas.statistikportal.de/_opendata2022.html, unzip and localise the LINREF files in one folder
# and adjust the path below.
# local functions
from utility_funcs import get_splitsuf

def main_acc_data_gen():
    #parameters
    Full = False
    Enc = True
    test_mode = False
    datasetrootname = 'v221217_NodeR0'
    split_strategy = 'yearly' #options: randomsamp, yearly, NBH, historical
    node_tolerance = 10
    node_radius = 0

    # set path and datasetname
    rootpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) #Abs path of AccIndex.git
    data_path = os.path.join(rootpath,'data')
    # datasetname prefix and suffixes
    regioncode = 'DSR' if test_mode else 'Berlin'
    suf = '_Full' if Full else '' 
    suf = suf+'_NoEnc' if not Enc else suf+''
    splitsuf = get_splitsuf(split_strategy)
    dataset_name = regioncode + "_" + datasetrootname + suf
    Path(os.path.join(data_path, dataset_name)).mkdir(parents=True, exist_ok=True)
    graphpath = os.path.join(data_path, dataset_name, "Graphs_split"+split_strategy+'_'+splitsuf)
    Path(graphpath).mkdir(parents=True, exist_ok=True)

    # Enter the edge and node attributes according encoding scheme
    # Full list of node attributes: 
    # fullattr_nodes= ['nr_accidents', 'x', 'y', 'street_count', 'highway']
    # Full list of edge attributes: 
    # fullattr_edges = ['nr_accidents', 'name', 'length','width','lanes','maxspeed', 'reversed','oneway','junction','access','bridge','service','tunnel','highway']
    if split_strategy in ['yearly', 'randomsamp', 'NBH']:
        attribute_dict = {"one_hot_encoding" : {"node" : ['highway'],
                                                "edge" : ['maxspeed', 'highway','tunnel','access','reversed','junction','service','oneway','bridge']},
                        "label_encoding" : {"node" : [],
                                            "edge" : []},
                        "binary_encoding" : {"node" : [],
                                            "edge" : []},
                        "float_encoding" : {"node" : [],
                                            "edge" : []},
                        "unencoded" : {"node" : [ 'street_count'],
                                        "edge" : [ 'length']}
                        }
    elif split_strategy == 'historical':
        print("historical TVT strategy")
        attribute_dict = { 'train' : {"one_hot_encoding" : {"node" : ['highway'],
                                                            "edge" : ['maxspeed', 'highway','tunnel','access','reversed','junction','service','oneway','bridge']},
                                    "label_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "binary_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "float_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "unencoded" : {"node" : ['nr_accident_18', 'street_count'],
                                                    "edge" : [ 'length']}
                                    },
                            'val' : {"one_hot_encoding" : {"node" : ['highway'],
                                                            "edge" : ['maxspeed', 'highway','tunnel','access','reversed','junction','service','oneway','bridge']},
                                    "label_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "binary_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "float_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "unencoded" : {"node" : ['nr_accident_19', 'street_count'],
                                                    "edge" : [ 'length']}
                                    },
                            'test' : {"one_hot_encoding" : {"node" : ['highway'],
                                                            "edge" : ['maxspeed', 'highway','tunnel','access','reversed','junction','service','oneway','bridge']},
                                    "label_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "binary_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "float_encoding" : {"node" : [],
                                                        "edge" : []},
                                    "unencoded" : {"node" : ['nr_accident_20', 'street_count'],
                                                    "edge" : [ 'length']}
                                    }
                        }


    create_accident_data()
    #print("accident data loaded")
    acc_data_all = pd.read_csv(os.path.join(data_path,"acc_berlin_2018_to_2022_clean.csv"))
    acc_data_all["decimal_data"] = acc_data_all["UJAHR"] + acc_data_all["UMONAT"]/12

    G_orig = create_osm_data(city_name="New York City",node_tolerance=node_tolerance, test = test_mode, Full=Full)

    if split_strategy in ['yearly', 'NBH']:
        split_yearly(G_orig, node_radius, Enc, acc_data_all, attribute_dict, graphpath, test_mode, split_strategy)
    elif split_strategy == 'randomsamp':
        split_randomsamp(G_orig, node_radius, Enc, acc_data_all, attribute_dict, graphpath, test_mode)
    elif split_strategy == 'historical':
        split_historical(G_orig, node_radius, Enc, acc_data_all, attribute_dict, graphpath, test_mode)	


def split_historical(G_orig, node_radius, Enc, acc_data_all, attribute_dict, graphpath, test_mode):
    '''
    Split the data into train, validation and test data based on the historical accident data
    Creates a graph for each split and encodes the data according to the encoding scheme.
    And saves the graphs as .pkl files in the graphpath folder.

    G_orig: original graph
    node_radius: radius of the node neighbourhood
    Enc: "True" if encoding is desired, "False" otherwise
    acc_data_all: dataframe with all accident data
    attribute_dict: dictionary with the attributes to be encoded
    graphpath: path to the graph
    test_mode: "True" if test mode is desired, "False" otherwise
    '''
    historical = True
    print("TVTsplit_historical starts!")

    # create train, validate and test data
    for split in ['train', 'val', 'test']:
        G = G_orig.copy() # strictly not necessary but for book keeping
        # acc_data_split_history = acc_data_all[(acc_data_all["decimal_data"]>=(period["history"]))]
        # acc_data_split_label = acc_data_all[(acc_data_all["decimal_data"]>=(period["label"]))]
        # Giving the data a nice treatment
        node_updated_G, accident_data_for_edges = add_attr_to_nodes(G, acc_data_all, acc_corr_factor=None, radius = node_radius, test = test_mode, historical = historical)
        G_with_localised_acc_attributes = add_attr_to_edges(node_updated_G, accident_data_for_edges, historical = historical)
        # Encoding the data
        if Enc:
            G_final = encoding_variables(G_with_localised_acc_attributes, attribute_dict, historical, split)
        else:
            G_final = G_with_localised_acc_attributes

        # Save the data
        # nx.write_gpickle(G_final, os.path.join(graphpath, split + ".gpickle"))
        fulldata_path = os.path.join(graphpath, split + ".pkl")
        pickle.dump(G_final, open(fulldata_path, "wb"))
        print("Saved " + split + " data to " + fulldata_path)


def split_randomsamp(G_orig, node_radius, Enc, acc_all, attribute_dict, graphpath, test_mode):
    '''
    Split the data into train, validation and test data based on a random sample
    Creates a graph for each split and encodes the data according to the encoding scheme.
    And saves the graphs as .pkl files in the graphpath folder.

    G_orig: original graph
    node_radius: radius of the node neighbourhood
    Enc: "True" if encoding is desired, "False" otherwise
    acc_all: dataframe with all accident data
    attribute_dict: dictionary with the attributes to be encoded
    graphpath: path to the graph
    test_mode: "True" if test mode is desired, "False" otherwise
    '''

    splits = {'train':0.50, 'val':0.25, 'test':0.25}

    acc_train =  pd.DataFrame(columns = acc_all.columns)
    acc_val = pd.DataFrame(columns = acc_all.columns)
    acc_test = pd.DataFrame(columns = acc_all.columns)
    for month in range(1,13):
        acc_all_m = acc_all[acc_all['UMONAT']==month]
        acc_m_train, acc_m_valtest = train_test_split(acc_all_m, test_size=splits['test']+splits['val'], shuffle=True)
        acc_m_val, acc_m_test =  train_test_split(acc_m_valtest, test_size=splits['test']/(splits['test']+splits['val']), shuffle=True)
        acc_train = pd.concat([acc_train, acc_m_train])
        acc_val = pd.concat([acc_val, acc_m_val])
        acc_test = pd.concat([acc_test, acc_m_test])

    accD = {'train':acc_train, 'val':acc_val, 'test':acc_test}

    num_years = 1 + acc_all['UJAHR'].max() - acc_all['UJAHR'].min()
    # assuming a homogeneous dist of accidents through the year, 
    # dividing the num. accidents by the following correction factors, should yield #acc/year 
    acc_corr_factor = {"train": num_years * splits['train'], 
                        "val": num_years * splits['val'], 
                        "test": num_years * splits['test']}

    for split, acc_split in accD.items():
        G = G_orig.copy() # strictly not necessary but for book keeping
        # Giving the data a nice treatment
        node_updated_G, accident_data_for_edges = add_attr_to_nodes(G, acc_split, acc_corr_factor[split], radius = node_radius, test = test_mode)
        G_with_localised_acc_attributes = add_attr_to_edges(node_updated_G, accident_data_for_edges, acc_corr_factor[split])
        if Enc:
            G_final = encoding_variables(G_with_localised_acc_attributes, attribute_dict)
        else:
            G_final = G_with_localised_acc_attributes

        fulldata_path = os.path.join(graphpath, split + '.pkl')
        pickle.dump(G_final, open(fulldata_path, 'wb'))
        print("Saved the full graph to drive:"+fulldata_path)
    

def split_yearly(G_orig, node_radius, Enc, acc_data_all, attribute_dict, graphpath, test_mode, split_strategy):
    '''
    Split the data into train, validation and test data based on a yearly split
    Creates a graph for each split and encodes the data according to the encoding scheme.
    And saves the graphs as .pkl files in the graphpath folder.
    
    G_orig: original graph
    node_radius: radius of the node neighbourhood
    Enc: "True" if encoding is desired, "False" otherwise
    acc_data_all: dataframe with all accident data
    attribute_dict: dictionary with the attributes to be encoded
    graphpath: path to the graph
    test_mode: "True" if test mode is desired, "False" otherwise
    split_strategy: "NBH" if neighbourhood split is desired, "YR" if yearly split is desired
    '''

    if split_strategy == 'NBH':
        TVT_split_years = {'nosplit': [2018,2019,2020,2021]}                                  
    else:
        #Enter years in each split
        TVT_split_years ={'train': [2018,2021],
                          'val': [2019,2020],
                          'test': [2021]} 

    for split, yearlist in TVT_split_years.items():
        G = G_orig.copy() # strictly not necessary but for book keeping
        
        acc_data_split = acc_data_all[acc_data_all['UJAHR'].isin(yearlist)]
        acc_corr_factor = len(yearlist)

        # Giving the data a nice treatment
        node_updated_G, accident_data_for_edges = add_attr_to_nodes(G, acc_data_split, acc_corr_factor, radius = node_radius, test = test_mode)
        G_with_localised_acc_attributes = add_attr_to_edges(node_updated_G, accident_data_for_edges, acc_corr_factor)
        if Enc:
            G_final = encoding_variables(G_with_localised_acc_attributes, attribute_dict)
        else: 
            G_final = G_with_localised_acc_attributes

        fulldata_path = os.path.join(graphpath, split + '.pkl')
        pickle.dump(G_final, open(fulldata_path, 'wb'))
        print("Saved the full graph to drive:"+fulldata_path)

#previously this was called split_yearly
def split_periods(G_orig, node_radius, Enc, acc_data_all, attribute_dict, graphpath, test_mode, split_strategy):
    '''
    Split the data into train, validation and test data based on a yearly split
    Creates a graph for each split and encodes the data according to the encoding scheme.
    And saves the graphs as .pkl files in the graphpath folder.

    :param G_orig: original graph
    :param node_radius: radius of the node neighbourhood
    :param Enc: "True" if encoding is desired, "False" otherwise
    :param acc_data_all: dataframe with all accident data
    :param attribute_dict: dictionary with the attributes to be encoded
    :param graphpath: path to the graph
    :param test_mode: "True" if test mode is desired, "False" otherwise
    :param split_strategy: "NBH" if neighbourhood split is desired, "YR" if yearly split is desired
    '''

    if split_strategy == 'NBH':
        train_validate_test_split_periods = {"nosplit" : {"start": {"year": 2018, "month":1}, "end_inclusive": {"year": 2021, "month": 12}}}
        acc_corr_factors = {"nosplit": 2}                             
    else:
        # Enter start and end of train, validate and test period
        train_validate_test_split_periods = {"train" : {"start": {"year": 2018, "month":1}, "end_inclusive": {"year": 2019, "month": 12}},
                                            "val" : {"start": {"year": 2020, "month":1}, "end_inclusive": {"year": 2021, "month": 12}},
                                            "test" : {"start": {"year": 2021, "month":1}, "end_inclusive": {"year": 2021, "month": 12}}}
        acc_corr_factors = {"train": 2, 
                           "val": 2, 
                           "test": 1}
    
    for split, period in train_validate_test_split_periods.items():
        G = G_orig.copy() # strictly not necessary but for book keeping
        acc_data_split = acc_data_all[(acc_data_all["decimal_data"]>=(period["start"]["year"]+period["start"]["month"]/12))  &
                                     (acc_data_all["decimal_data"]<=(period["end_inclusive"]["year"]+period["end_inclusive"]["month"]/12))]
        acc_corr_factor = acc_corr_factors[split]

        # Giving the data a nice treatment
        node_updated_G, accident_data_for_edges = add_attr_to_nodes(G, acc_data_split, acc_corr_factor, radius = node_radius, test = test_mode)
        G_with_localised_acc_attributes = add_attr_to_edges(node_updated_G, accident_data_for_edges, acc_corr_factor)
        if Enc:
            G_final = encoding_variables(G_with_localised_acc_attributes, attribute_dict)
        else: 
            G_final = G_with_localised_acc_attributes

        fulldata_path = os.path.join(graphpath, split + '.pkl')
        pickle.dump(G_final, open(fulldata_path, 'wb'))
        print("Saved the full graph to drive:"+fulldata_path)


def create_accident_data():
    '''
    Cleans the accident data and saves it as a .csv file
    '''
    total_df = pd.DataFrame()

    for y in [2016,2017,2018,2019,2020,2021]:
        print(y)

        df = pd.read_csv("../data/Berlin_accident_raw/collection/Unfallorte" + str(y) + "_LinRef.csv", sep=";")

        df.rename(columns = {"OBJECTID_1":"OBJECTID"}, inplace=True)
        df.rename(columns = {"USTRZUSTAND":"STRZUSTAND"}, inplace=True)
        df.rename(columns = {"IstSonstig":"IstSonstige"}, inplace=True)
        df.rename(columns = {"LICHT":"ULICHTVERH"}, inplace=True)
        if "UIDENTSTLA" in df.columns:
            df["UIDENTSTLAE"] = df["UIDENTSTLA"]
            df.drop("UIDENTSTLA", axis =1, inplace=True)

        for c in ['IstGkfz' ]:
            if c not in df.columns:
                df[c] = None
        df.rename({ 'XGCSWGS84':'lon', 'YGCSWGS84':'lat'}, inplace=True, axis=1)

        df.loc[:, 'lat'] = df.loc[:, 'lat'].apply(lambda x: str(x).replace(',','.'))
        df.loc[:, 'lon'] = df.loc[:, 'lon'].apply(lambda x: str(x).replace(',','.'))
        df.loc[:, 'lon'] = df.loc[:, 'lon'].astype(float)
        df.loc[:, 'lat'] = df.loc[:, 'lat'].astype(float)

        print(df.shape)
        print(df.columns)

        total_df = pd.concat([total_df,df], axis=0)

    total_df.to_csv("total.csv")

    berlin_df = total_df[total_df["ULAND"]==0]
    berlin_df.to_csv("../data/acc_berlin_2018_to_2022_clean.csv")

    return berlin_df


def create_osm_data(city_name="Berlin", node_tolerance=10, test=False, Full=False):
    '''
    :param city_name: name of the city for which the graph is created
    :param node_tolerance: tolerance for node merging
    :param test: if True, only a small part of the city is loaded
    :param  Full: if True, the full graph is loaded
    :return: the graph
    '''

    print("loading Graph")
    if test:
        print("### Working in Test mode ###")
        DSR_lat, DSR_lon = 52.500539, 13.334758
        north, south, east, west = DSR_lat + 0.01, DSR_lat - 0.01, DSR_lon + 0.02, DSR_lon - 0.02
        # create network from that bounding box
        G = ox.graph_from_bbox(north, south, east, west, network_type="drive_service")

    else:
        G = ox.graph_from_place(city_name, network_type="drive")
    G = ox.project_graph(G)
    print("consolidate intersections")
    G = ox.consolidate_intersections(G, rebuild_graph=True, tolerance=node_tolerance, dead_ends=False)
    # project back to geo coordinate system
    G = ox.project_graph(G, to_crs="EPSG:4326")

    # We need this block only for csv-file generation
    # For the G to be used for neighborhood generation, things will be edited during encoding anyway
    if Full:
        for s, r, e in G.edges(data=True):
            #e.pop('width',None)
            for k in ['tunnel', 'service', 'reversed', 'v_original', 'bridge', 'maxspeed', 'name', 'access', 'junction', 'highway', 'oneway', 'u_original', 'osmid', 'length', 'geometry', 'accidents_list','width',  'lanes']:
                if k not in e.keys():
                    e[k] = []
        
        for i, n in G.nodes(data=True):
            #e.pop('width',None)
            for k in ['accidents_list', 'osmid', 'highway', 'lat', 'lon']:
                if k not in n.keys():
                    n[k] = []
        
        #get and attach pop density and area use attributes
        #todo

        #node_updated_G, accident_data_for_edges = add_attr_to_nodes(G, acc_data_split, train_accident_correction_factor[split], radius = node_radius, test = test_mode)
        #G_with_localised_acc_attributes = add_attr_to_edges(node_updated_G, accident_data_for_edges, train_accident_correction_factor[split])

    return G


def add_attr_to_nodes(G, df, acc_corr_factor=None, radius = 20, test = False, historical = False):
    '''
    This function finds the nearest node for each accident and assigns the accident information to the node in the graph
    :param G: The graph
    :param df: dataframe of attributes
    :param acc_corr_factor: The correction factor for the accident data
    :param radius: The radius for the nearest node search
    :param test: boolean if true, the function will only create a small graph for testing purposes
    :param historical: boolean if true, the function will assign the accident information to the nodes for each timestamp
    :return: The graph with the accident information assigned to the nodes and the accident data for the edges'''

    if test:
        DSR_lat, DSR_lon = 52.500539, 13.334758
        north, south, east, west = DSR_lat + 0.01, DSR_lat - 0.01, DSR_lon + 0.02, DSR_lon - 0.02
        df = df[(df['lat']>south) & (df['lat']<north) & (df['lon']>west) & (df['lon']<east)]

    lon_list, lat_list = list(df['lon']), list(df['lat'])

    print("Find nearest nodes")

    nodes, dists = ox.nearest_nodes(G, lon_list, lat_list, return_dist=True)

    # Identify the accident-node combinations that are close enough together
    within_node_radius_mask = [True if d < radius else False for d in dists]
    
    node_accs = [b for a, b in zip(within_node_radius_mask, nodes) if a]

    if historical:
        G = add_histacc_to_G_nodes(G, df, node_accs, within_node_radius_mask)
    else:
        G = add_acc_to_G_nodes(G, df, within_node_radius_mask, node_accs, acc_corr_factor)

    # return updated Graph and dataframe of the remaining, unassigned accidents
    return G, df[[not e for e in within_node_radius_mask]]


def add_attr_to_edges(G, df, acc_corr_factor=None, historical=False):
    '''
    Find the nearest edge for each accident and add the accident to the edge
    :param G: Graph
    :param df: dataframe of attributes
    :param acc_corr_factor: correction factor for train accidents
    :param historical: boolean, if true, the function will add the accidents to the edges  for each timestamp
    :return: The graph with the accident information assigned to the edges
    '''

    print("Find nearest edges")

    lon_list, lat_list = list(df['lon']), list(df['lat'])

    edges = ox.nearest_edges(G, lon_list, lat_list, return_dist=False)
    # put data of accidents in list of dictionary
    accident_list = [{"accident" : v } for v in df.T.to_dict().values()]

    if historical:
        G = add_histacc_to_G_edges(G, df, edges)
    else:
        G = add_acc_to_G_edges(G, df, edges, acc_corr_factor)

    return G

    
def add_acc_to_G_nodes(G, df, within_node_radius_mask, node_accs, acc_corr_factor):
    '''
    Add the accident information to the nodes of the graph
    :param G: Graph
    :param df: dataframe of attributes
    :param within_node_radius_mask: mask for accidents that are close enough to a node
    :param node_accs: list of nodes that have an accident close enough
    :param acc_corr_factor: correction factor for train accidents
    :return: The graph with the accident information assigned to the nodes
    '''
        
    # put data of accidents in list of dictionary
    node_accident_list = [{"accident" : v } for v in df[within_node_radius_mask].T.to_dict().values()]

    # combine node information (nodes ids) with the accident data
    node_accident_connection_dict = {}
    for i in zip(node_accs, node_accident_list):
        node_accident_connection_dict.setdefault(i[0],[]).append(i[1])

    # Update the nodes of the graph with accident data
    for node, accident in node_accident_connection_dict.items():
        nx.set_node_attributes(G,{node:{'accidents_list':accident}})

    for node, accident in node_accident_connection_dict.items():
        nx.set_node_attributes(G,{node:{'nr_accidents':len(accident)/acc_corr_factor}})

    # Set accidents_list and nr_accidents for nodes that have no accidents
    for i, n in G.nodes(data=True):
        if "accidents_list" not in n.keys():
            n['accidents_list'] = []
        if "nr_accidents" not in n.keys():
            n["nr_accidents"] = 0
    
    return G


def add_acc_to_G_edges(G, df, edges, acc_corr_factor):
    '''
    Add the accident information to the edges of the graph
    :param G: Graph
    :param df: dataframe of attributes
    :param edges: list of edges
    :param acc_corr_factor: correction factor for train accidents
    :return: The graph with the accident information assigned to the edges
    '''

    # put data of accidents in list of dictionary
    accident_list = [{"accident" : v } for v in df.T.to_dict().values()]

    # combine edge information (edge_ids ids) with the accident data
    edge_accident_connection_dict = {}
    for i in zip(edges, accident_list):
        edge_accident_connection_dict.setdefault(i[0],[]).append(i[1])

    # Update the edge of the graph with accident data
    for edge, accident in edge_accident_connection_dict.items():
        nx.set_edge_attributes(G,{edge:{'accidents_list':accident}})

    for edge, accident in edge_accident_connection_dict.items():
        nx.set_edge_attributes(G,{edge:{'nr_accidents':len(accident)/acc_corr_factor}})

    # Set accidents_list and nr_accidents for edges that have no accidents
    for s, r, e in G.edges(data=True):
        if "accidents_list" not in e.keys():
            e["accidents_list"] = []
        if "nr_accidents" not in e.keys():
            e["nr_accidents"] = 0

    return G


def add_histacc_to_G_nodes(G, df, node_accs, within_node_radius_mask):
    '''
    Add the accident information to the nodes of the graph for each year
    :param G: Graph
    :param df: dataframe of attributes
    :param node_accs: list of nodes that have an accident close enough
    :param within_node_radius_mask: mask for accidents that are close enough to a node
    :return: The graph with the accident information assigned to the nodes
    '''

    print("Historical data")
    # put data of accidents in list of dictionary
    #split the data according to year
    #Put the data in a list of dictionaries
    #may be we need to add .loc for the boolean mask df.loc[within_node_radius_mask].loc[df.UJAHR == 2018]
    node_accident_list_2018 = [{"accident_18" : v } for v in df[within_node_radius_mask][df.UJAHR == 2018].T.to_dict().values()] 
    node_accident_list_2019 = [{"accident_19" : v } for v in df[within_node_radius_mask][df.UJAHR == 2019].T.to_dict().values()]
    node_accident_list_2020 = [{"accident_20" : v } for v in df[within_node_radius_mask][df.UJAHR == 2020].T.to_dict().values()]
    node_accident_list_2021 = [{"accident_21" : v } for v in df[within_node_radius_mask][df.UJAHR == 2021].T.to_dict().values()]

    
    #combine node information (nodes ids) with the accident data
    node_accident_connection_dict_2018 = {}
    for i in zip(node_accs, node_accident_list_2018):
        node_accident_connection_dict_2018.setdefault(i[0],[]).append(i[1])
    
    node_accident_connection_dict_2019 = {}
    for i in zip(node_accs, node_accident_list_2019):
        node_accident_connection_dict_2019.setdefault(i[0],[]).append(i[1])

    node_accident_connection_dict_2020 = {}
    for i in zip(node_accs, node_accident_list_2020):
        node_accident_connection_dict_2020.setdefault(i[0],[]).append(i[1])
    
    node_accident_connection_dict_2021 = {}
    for i in zip(node_accs, node_accident_list_2021):
        node_accident_connection_dict_2021.setdefault(i[0],[]).append(i[1])
    
    #Update the nodes of the graph with accident data
    for node, accident in node_accident_connection_dict_2018.items():
        nx.set_node_attributes(G,{node:{'accidents_list_18':accident}})
    
    for node, accident in node_accident_connection_dict_2018.items():
        nx.set_node_attributes(G,{node:{'nr_accidents_18':len(accident)}})
    
    for node, accident in node_accident_connection_dict_2019.items():
        nx.set_node_attributes(G,{node:{'accidents_list_19':accident}})
    
    for node, accident in node_accident_connection_dict_2019.items():
        nx.set_node_attributes(G,{node:{'nr_accidents_19':len(accident)}})
    
    for node, accident in node_accident_connection_dict_2020.items():
        nx.set_node_attributes(G,{node:{'accidents_list_20':accident}})
    
    for node, accident in node_accident_connection_dict_2020.items():
        nx.set_node_attributes(G,{node:{'nr_accidents_20':len(accident)}})
    
    for node, accident in node_accident_connection_dict_2021.items():
        nx.set_node_attributes(G,{node:{'accidents_list_21':accident}})
    
    for node, accident in node_accident_connection_dict_2021.items():
        nx.set_node_attributes(G,{node:{'nr_accidents_21':len(accident)}})

    # Set accidents_lists and nr_accidents for nodes that have no accidents
    for node in G.nodes():
        if 'accidents_list_18' not in G.nodes[node]:
            nx.set_node_attributes(G,{node:{'accidents_list_18':[]}})
            nx.set_node_attributes(G,{node:{'nr_accidents_18':0}})
        if 'accidents_list_19' not in G.nodes[node]:
            nx.set_node_attributes(G,{node:{'accidents_list_19':[]}})
            nx.set_node_attributes(G,{node:{'nr_accidents_19':0}})
        if 'accidents_list_20' not in G.nodes[node]:
            nx.set_node_attributes(G,{node:{'accidents_list_20':[]}})
            nx.set_node_attributes(G,{node:{'nr_accidents_20':0}})
        if 'accidents_list_21' not in G.nodes[node]:
            nx.set_node_attributes(G,{node:{'accidents_list_21':[]}})
            nx.set_node_attributes(G,{node:{'nr_accidents_21':0}})
    return G
    
def add_histacc_to_G_edges(G, df, edges):
    '''
    Add historical accident data to the edges of the graph
    :param G: graph
    :param df: dataframe with accident data
    :param edges: list of edges
    :return: graph with historical accident data added to the edges
    '''
    print("Historical data")
    # put data of accidents in list of dictionary
    accident_list_2018 = [{"accident_18" : v } for v in df[df.UJAHR == 2018].T.to_dict().values()]
    accident_list_2019 = [{"accident_19" : v } for v in df[df.UJAHR == 2019].T.to_dict().values()]
    accident_list_2020 = [{"accident_20" : v } for v in df[df.UJAHR == 2020].T.to_dict().values()]
    accident_list_2021 = [{"accident_21" : v } for v in df[df.UJAHR == 2021].T.to_dict().values()]

    # combine edge information (edge_ids ids) with the accident data
    edge_accident_connection_dict_2018 = {}
    for i in zip(edges, accident_list_2018):
        edge_accident_connection_dict_2018.setdefault(i[0],[]).append(i[1])
    
    edge_accident_connection_dict_2019 = {}
    for i in zip(edges, accident_list_2019):
        edge_accident_connection_dict_2019.setdefault(i[0],[]).append(i[1])
    
    edge_accident_connection_dict_2020 = {}
    for i in zip(edges, accident_list_2020):
        edge_accident_connection_dict_2020.setdefault(i[0],[]).append(i[1])

    edge_accident_connection_dict_2021 = {}
    for i in zip(edges, accident_list_2021):
        edge_accident_connection_dict_2021.setdefault(i[0],[]).append(i[1])

    # Update the edge of the graph with accident data
    for edge, accident in edge_accident_connection_dict_2018.items():
        nx.set_edge_attributes(G,{edge:{'accidents_list_18':accident}})
    
    for edge, accident in edge_accident_connection_dict_2018.items():
        nx.set_edge_attributes(G,{edge:{'nr_accidents_18':len(accident)}})
    
    for edge, accident in edge_accident_connection_dict_2019.items():
        nx.set_edge_attributes(G,{edge:{'accidents_list_19':accident}})

    for edge, accident in edge_accident_connection_dict_2019.items():
        nx.set_edge_attributes(G,{edge:{'nr_accidents_19':len(accident)}})
    
    for edge, accident in edge_accident_connection_dict_2020.items():
        nx.set_edge_attributes(G,{edge:{'accidents_list_20':accident}})
    
    for edge, accident in edge_accident_connection_dict_2020.items():
        nx.set_edge_attributes(G,{edge:{'nr_accidents_20':len(accident)}})

    for edge, accident in edge_accident_connection_dict_2021.items():
        nx.set_edge_attributes(G,{edge:{'accidents_list_21':accident}})
    
    for edge, accident in edge_accident_connection_dict_2021.items():
        nx.set_edge_attributes(G,{edge:{'nr_accidents_21':len(accident)}})
    
    # Set accidents_list and nr_accidents for edges that have no accidents
    for s, r, e  in G.edges(data=True):
        if "accidents_list_18" not in e.keys():
            e['accidents_list_18'] = []
        if "nr_accidents_18" not in e.keys():
            e["nr_accidents_18"] = 0
        if "accidents_list_19" not in e.keys():
            e['accidents_list_19'] = []
        if "nr_accidents_19" not in e.keys():
            e["nr_accidents_19"] = 0
        if "accidents_list_20" not in e.keys():
            e['accidents_list_20'] = []
        if "nr_accidents_20" not in e.keys():
            e["nr_accidents_20"] = 0
        if "accidents_list_21" not in e.keys():
            e['accidents_list_21'] = []
        if "nr_accidents_21" not in e.keys():
            e["nr_accidents_21"] = 0

    return G


def encoding_variables(G, attribute_dict, historical=False, split=None):
    '''
    Encode categorical variables
    :param G: graph
    :param attribute_dict: dictionary with attributes
    :param historical: boolean, if historical data is used
    :param split: string, train, val or test
    :return: graph with encoded variables
    '''
    
    if historical:
        print("Encoding historical variables")
        # assign attributes to the Graph object that are either original from the data, or are created during encoding.
        # Currently only One hot encoding is implemented.
        #assign attribute according to the split
        edge_attributes_list = []
        node_attributes_list = []
        edge_attributes_list = edge_attributes_list + attribute_dict[split]["unencoded"]["edge"]
        node_attributes_list = node_attributes_list + attribute_dict[split]["unencoded"]["node"]

        print("Encoding edge and node variables")
        for edge_var in attribute_dict[split]["one_hot_encoding"]["edge"]:
            G, new_attributes = encode_categorical_variables(G, edge_var, 'edge')
            edge_attributes_list = edge_attributes_list + new_attributes

        for node_var in attribute_dict[split]["one_hot_encoding"]["node"]:
            G, new_attributes = encode_categorical_variables(G, node_var, 'node')
            node_attributes_list = node_attributes_list + new_attributes
        
        G.edge_attributes_list = edge_attributes_list
        G.node_attributes_list = node_attributes_list
        
    
    else:
        print("Encoding variables")    
        # assign attributes to the Graph object that are either original from the data, or are created during encoding.
        # Currently only One hot encoding is implemented.
        edge_attributes_list = []
        node_attributes_list = []
        edge_attributes_list = edge_attributes_list + attribute_dict["unencoded"]["edge"]
        node_attributes_list = node_attributes_list + attribute_dict["unencoded"]["node"]

        print("Encoding edge and node variables")
        for edge_var in attribute_dict["one_hot_encoding"]["edge"]:
            G, new_attributes = encode_categorical_variables(G, edge_var, 'edge')
            edge_attributes_list = edge_attributes_list + new_attributes

        for node_var in attribute_dict["one_hot_encoding"]["node"]:
            G, new_attributes = encode_categorical_variables(G, node_var, 'node')
            node_attributes_list = node_attributes_list + new_attributes

        G.edge_attributes_list = edge_attributes_list
        G.node_attributes_list = node_attributes_list

    return G
    
if __name__ == '__main__':
    main_acc_data_gen()