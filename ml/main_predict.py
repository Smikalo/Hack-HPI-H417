
import os

from source import neighbourhood as nbh
from source import viz
from model_funcs_accind import build_model_accind
import torch
from data_funcs import parameters
from torch_geometric.loader import DataLoader
from functools import reduce
import pickle
import time
import pandas as pd
from shapely import wkt
from utils.utility_funcs import get_datetime_string

def main_predict(G, model_path, density_df):

    #model_path = os.path.join(NBHdatapath, "gn_model.pt")
    state_dict = torch.load(model_path)
    coords_standard_scaler_path = os.path.join(NBHdatapath,"coords_standard_scaler.pkl")
    density_minmax_scaler_path = os.path.join(NBHdatapath,"density_minmax_scaler.pkl")
    coords_standard_scaler = pickle.load(open(coords_standard_scaler_path,'rb'))
    density_minmax_scaler = pickle.load(open(density_minmax_scaler_path,'rb'))

    nest_size = 5
    orig = [40.709327, -73.819348]
    dest = [40.753679, -73.898784]
    print('orig', orig)
    print('dest', dest)
    path = nbh.shortest_path_on_map(G, orig, dest)
    print('path', path[0])
    
    path_node_edge_pairs = nbh.get_node_edge_pairs_from_path(path[0]) #take the first path: for the time being
    acc_dict_nodes = dict()
    acc_dict_edges = dict()

    for i, node_edge_pairs in enumerate(path_node_edge_pairs):
        node = node_edge_pairs[0]
        edge = node_edge_pairs[1]
        nest_of_nbh, _ , _ = nbh.nest_of_neighbourhoods_aroud_node_edge(G, node, edge, hop=3, num_nbh=nest_size+1)
        if i == 0:
            #single datapoint for extracting the data parameters
            DSpars = parameters(nbh.attr_aggregation_cleaning_converting_N([nest_of_nbh[0], node, edge], density_df)) #data parameters TODO: find a way to call the parameters from the model/effeitiantly some other way
            #loading the model
            model, _ , loss_funtion, _ = build_model_accind(DSpars, report=False)
            model.load_state_dict(state_dict)
            model.eval()
            print('model loaded!')

        nest_of_nbh_enriched = []
        for i in range(1, len(nest_of_nbh)):
            nest_of_nbh_enriched.append(nbh.attr_aggregation_cleaning_converting_N([nest_of_nbh[i], node, edge], density_df))
        density, coords = nbh.scaling_collect(nest_of_nbh_enriched)
        nest_of_nbh_enriched = nbh.scaling_transform(nest_of_nbh_enriched, density, coords, density_minmax_scaler, coords_standard_scaler)
        data_collection = DataLoader(nest_of_nbh_enriched, batch_size=len(nest_of_nbh_enriched))
        for data in data_collection:
            with torch.inference_mode():
                out = model(data.x, data.edge_attr, data.u, data.edge_index, data.n_pred_idx, data.e_pred_idx, data.batch)
                print('Output', out)
                print('Target', data.y.to(torch.float32))
                print('Error', loss_funtion(out, data.y.to(torch.float32)))
                print('------------------------------------')
        out_put = torch.mean(out,0)
        #out_put = reduce(lambda x,y: x+y,out_put.tolist())

        acc_dict_nodes[node] = out_put[0]
        acc_dict_edges[edge] = out_put[1]
        print('------------------------------------')

    # viz.plot_NEST_nbh_node_edge_on_map(G,  N_lst=lst, node=node, edge=edge, show=True)#, rootfolder=None, rootfname=None)
    # viz.plot_nbh_node_edge_on_map(G, lst[0], node, edge)

    print('done!')
    
    return path[0], acc_dict_nodes, acc_dict_edges

if __name__ == "__main__":
    print("Starting....................")
    start_time = time.time()
    use_case = 'AccInd'
    verbosetest = False
    numepochs = 50
    datasetrootname = "v221217_NodeR0"
    split_strategy = 'yearly' #options: randomsamp, yearly. # TODO: historical
    splitsuf_yearly = 'T1821_V1920_T21'
    splitsuf_random = 'T07_V015_T015'
    n = 2 #number of hops used to generate the neighborhoods
    num_nbh = 200 #number of neighbourhoods to generate
    num_nbh_VT_factor = 0.5 #0.2
    modelrootname = 'DO025_lr0005_20epochs.pt'

    # set paths and datasetname
    dataset_name = "Berlin_" + datasetrootname
    dataset_version = datasetrootname.split('_')[-1]
    rootpath = os.path.dirname(os.path.realpath(__file__)) #Abs path of AccIndex.git
    data_path = os.path.join(rootpath,'data')
    splitsuf = splitsuf_random if split_strategy == 'randomsamp' else splitsuf_yearly
    graphdatapath = os.path.join(data_path, dataset_name, "Graphs_split"+split_strategy+'_'+splitsuf)

    num_nbh_dict = {"train" : 60000,#num_nbh,
                    "val" : 15000, #int(num_nbh*num_nbh_VT_factor),
                    "test" : 15000 #1#int(num_nbh*num_nbh_VT_factor)
                    }
    splitsizestr = ''.join([key+str(int(val)) for key,val in num_nbh_dict.items()])
    NBHdatapath=os.path.join(graphdatapath, 'nbhs_graphs_nhop%d'%(n),splitsizestr)

    # set path and datasetname
    rootpath = os.path.dirname(os.path.realpath(__file__)) #Abs path of AccIndex.git
    data_path = os.path.join(rootpath,'data')
    dataset_path = os.path.join(data_path, dataset_name)
    #NBHdatapath = os.path.join(dataset_path, 'nbhs_graphs_n%d_numnbh%d'%(n,num_nbh))
    G = pickle.load(open(os.path.join(graphdatapath, "test.pkl"), 'rb'))
    print('G loaded')

    # Density and urban utilisation data downloaded via QGIS from https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s06_06ewdichte2021
    # File can be found on Dropbox
    population_density_data = 'Berlin_pop_density.csv'
    density_df = pd.read_csv( 'data/' + population_density_data)
    #density_df['geometry'] = density_df['geometry'].apply(wkt.loads)
    date_string, time_string = get_datetime_string()
    model_name = modelrootname# + '_' + date_string + '_' + time_string
    model_folder_path = os.path.join(NBHdatapath, 'models')
    model_rootpath = os.path.join(model_folder_path, model_name)# + '_' + str(numepochs)+'epochs')
    model_path = model_rootpath# + ".pt"

    # Call the predict function
    path, acc_dict_nodes, acc_dict_edges= main_predict(G, model_path, density_df)
    print("--- %s seconds ---" % (time.time() - start_time))

    viz.plot_path(G, path, acc_dict_nodes, acc_dict_edges)