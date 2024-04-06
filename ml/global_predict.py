#libraries
import os
import numpy as np
import pandas as pd
import pickle
from shapely import wkt
import time
import torch
from torch_geometric.loader import DataLoader
#local functions
from source import neighbourhood as nbh
from model_funcs_accind import build_model_accind
from data_funcs import parameters
from utils.utility_funcs import get_splitsuf

report_freq = 10
batchsize = 100
TESTMODE = False
if TESTMODE:
    report_freq = 5
    batchsize = 10
    testsize = 25

def main_predict(G, model_path, model_name, density_df):

    prediction_path = os.path.join(model_path,'predictions')
    os.mkdir(prediction_path) if not os.path.exists(prediction_path) else print(f'Prediction folder exists:{prediction_path}')
    pred_fname = os.path.join(prediction_path, "global_edge_preds.csv")

    coords_standard_scaler_path = os.path.join(NBHdatapath,"coords_standard_scaler.pkl")
    density_minmax_scaler_path = os.path.join(NBHdatapath,"density_minmax_scaler.pkl")
    coords_standard_scaler = pickle.load(open(coords_standard_scaler_path,'rb'))
    density_minmax_scaler = pickle.load(open(density_minmax_scaler_path,'rb'))

    G_node_edge_list = [(e[0], (e[0],e[1]),e[2]) for e in G.edges(data=True)]
    state_dict = torch.load(os.path.join(model_path,model_name))
    model = get_model(state_dict, 
                      G_node_edge_list[0], 
                      density_df)
    
    if (TESTMODE):
        G_node_edge_list = G_node_edge_list[0:testsize]

    pred_subfnames = predict_acc_for_node_and_edge_list_batchloop(
                        G_node_edge_list, 
                        model, 
                        density_df,
                        density_minmax_scaler, 
                        coords_standard_scaler,
                        pred_fname,
                        batchsize)
    
    append_csv_tables(pred_subfnames, pred_fname)

    print(f'Saved predictions in: {pred_fname}')

def append_csv_tables(pred_subfnames, pred_fname):

    print(f'Concatanating {len(pred_subfnames)} tables:')
    edge_preds_df = pd.read_csv(pred_subfnames[0])
    print(' Table index: 0', end = ' ')
    for t in range(1,len(pred_subfnames)):
        print(f' {t}', end = ' ')
        edge_preds_df_sub = pd.read_csv(pred_subfnames[t])
        edge_preds_df = pd.concat([edge_preds_df, edge_preds_df_sub], ignore_index=True)
    #edge_preds_df.reset_index(drop=True)
    print(' Done. Writing')
    edge_preds_df.to_csv(pred_fname, index=True)
    
def get_G_node_edge_sublist(G_node_edge_list, batchsize, len_preds):
    
    num_batches = int(np.ceil(len_preds/batchsize))

    G_node_edge_sublists = [None] * num_batches
    start_indices = np.arange(0, len_preds, step=batchsize)

    for l, G_node_edge_sublist in enumerate(G_node_edge_sublists):
        start_index = start_indices[l]
        end_index = min(len_preds, start_index+batchsize)
        G_node_edge_sublists[l] = G_node_edge_list[start_index:end_index]

    return G_node_edge_sublists, start_indices

def predict_acc_for_node_and_edge_list_batchloop(G_node_edge_list, model, density_df, density_minmax_scaler, coords_standard_scaler, pred_fname, batchsize):

    #split the list into sublists
    len_preds = len(G_node_edge_list)
    G_node_edge_sublists, start_indices = get_G_node_edge_sublist(G_node_edge_list,batchsize, len_preds)
    
    pred_subfnames = [None] * len(G_node_edge_sublists)
    print(f'Total of {len_preds} prediction tasks will be performed in {len(G_node_edge_sublists)} batches')
    for l,G_node_edge_sublist in enumerate(G_node_edge_sublists):
        start_index = start_indices[l]
        end_index = min(start_indices[l]+batchsize,len_preds)
        print(f'Batch {l+1}: ({start_index}-{end_index})')
        #filename
        pred_subfnames[l] = pred_fname.split('.csv')[0] + '_' + str(l)+'_'+ str(start_index) + '-' + str(end_index) + '.csv'
        if os.path.exists(pred_subfnames[l]):
            print(f' predictions appear to already exist in: {os.path.basename(pred_subfnames[l])}')
        else:
            acc_dict_edges, acc_dict_nodes = predict_acc_for_node_and_edge_list(
                                                G_node_edge_sublist, 
                                                model, density_df, 
                                                density_minmax_scaler, 
                                                coords_standard_scaler)
            edge_preds_df = pd.DataFrame(acc_dict_edges)
            edge_preds_df.to_csv(pred_subfnames[l],index=False)
            print(f' Wrote table: {os.path.basename(pred_subfnames[l])}')
    
    return(pred_subfnames)

def predict_acc_for_node_and_edge_list(G_node_edge_list, model, density_df, density_minmax_scaler, coords_standard_scaler):
    len_preds = len(G_node_edge_list)
    acc_dict_nodes = dict()
    acc_dict_edges = dict()
    node_list = [None]*len_preds
    edge_list = [None]*len_preds
    node_mean_out_collection = [None]*len_preds
    edge_std_collection = [None]*len_preds
    edge_mean_out_collection = [None]*len_preds
    start_lat_list = [None]*len_preds
    start_lon_list = [None]*len_preds
    end_lat_list = [None]*len_preds
    end_lon_list = [None]*len_preds
    print (f' Total prediction tasks: {len_preds}')
    start_time = time.time()

    for i, node_edge_pairs, in enumerate(G_node_edge_list):
        if (i+1)%report_freq == 0:
            print(f"  {i+1} done. Elapsed time: %s seconds" % (time.time() - start_time))
            start_time = time.time()
        
        node = node_edge_pairs[0]
        edge = node_edge_pairs[1]

        # make the prediction
        out = predict_acc_for_node_and_edge(node,
                                            edge,
                                            model,
                                            density_df, 
                                            density_minmax_scaler, 
                                            coords_standard_scaler, 
                                            nest_size=5, 
                                            nhop=3)

        out_put = torch.mean(out,0)
        out_std = torch.std(out,0)
        edge_std_collection[i] = float(out_std[1])
        node_mean_out_collection[i] = float(out_put[0])
        edge_mean_out_collection[i] = float(out_put[1])
        edge_list[i] = edge
        node_list[i] = node
        # the ordering of ["geometry"].boundary.bounds is arbitrary, but in the global prediction that doenst matter
        if len(node_edge_pairs[2]["geometry"].boundary.bounds) > 0:
            start_lat_list[i] = node_edge_pairs[2]["geometry"].boundary.bounds[1]
            start_lon_list[i] = node_edge_pairs[2]["geometry"].boundary.bounds[0]
            end_lat_list[i] = node_edge_pairs[2]["geometry"].boundary.bounds[3]
            end_lon_list[i] = node_edge_pairs[2]["geometry"].boundary.bounds[2]
        else: 
            # there are cases where the boundary.bounds is empty, for instance when the src and dest nodes are the same:
            # node: 13927
            # edge: (13927, 13927)
            # which may be caused by the Graph simplification. In any case, the lats/lons are not relevant for such cases:
            start_lat_list[i] = np.nan
            start_lon_list[i] = np.nan
            end_lat_list[i] = np.nan
            end_lon_list[i] = np.nan
        
    #print(edge_std_collection)
    #print(edge_mean_out_collection)

    #print(' creating table')
    acc_dict_nodes[node] = out_put[0] #Onur: we don't seem to use this?
    acc_dict_edges["node"] = node_list
    acc_dict_edges["edge"] = edge_list
    acc_dict_edges["predicted_edge_acc"] = edge_mean_out_collection
    acc_dict_edges["std_of_edge_pred"] = edge_std_collection
    acc_dict_edges["predicted_node_acc"] = node_mean_out_collection
    acc_dict_edges["start_lat"] = start_lat_list
    acc_dict_edges["start_lon"] = start_lon_list
    acc_dict_edges["end_lat"] = end_lat_list
    acc_dict_edges["end_lon"] = end_lon_list
    
    return acc_dict_edges, acc_dict_nodes

def predict_acc_for_node_and_edge (node, edge, model, density_df, density_minmax_scaler, coords_standard_scaler, nest_size=5, nhop=3):
    nest_of_nbh, _ , _ = nbh.nest_of_neighbourhoods_aroud_node_edge(G, node, edge, hop=nhop, num_nbh=nest_size+1)
    nest_of_nbh_enriched = []
    for n in range(1, len(nest_of_nbh)):
        nest_of_nbh_enriched.append(nbh.attr_aggregation_cleaning_converting_N([nest_of_nbh[n], node, edge], density_df))
    density, coords = nbh.scaling_collect(nest_of_nbh_enriched)
    nest_of_nbh_enriched = nbh.scaling_transform(nest_of_nbh_enriched, density, coords, density_minmax_scaler, coords_standard_scaler)
    data_collection = DataLoader(nest_of_nbh_enriched, batch_size=len(nest_of_nbh_enriched))
    for data in data_collection:
        with torch.inference_mode():
            out = model(data.x, data.edge_attr, data.u, data.edge_index, data.n_pred_idx, data.e_pred_idx, data.batch)
    return out

def get_model(state_dict, node_edge_pairs, density_df):
        node = node_edge_pairs[0]
        edge = node_edge_pairs[1]
        nest_of_nbh, _ , _ = nbh.nest_of_neighbourhoods_aroud_node_edge(G, node, edge, hop=3, num_nbh=1)
        #single datapoint for extracting the data parameters
        DSpars = parameters(nbh.attr_aggregation_cleaning_converting_N([nest_of_nbh[0], node, edge], density_df)) #data parameters TODO: find a way to call the parameters from the model/effeitiantly some other way
        #loading the model
        model, _, loss_function, _ = build_model_accind(DSpars, report=False)
        model.load_state_dict(state_dict)
        model.eval()
        print('model loaded!')
        return(model)

if __name__ == "__main__":
    print("Starting....................")
    use_case = 'AccInd'
    verbosetest = False
    numepochs = 50
    datasetrootname = "v221217_NodeR0_nhop2"
    split_strategy = 'yearly' #options: randomsamp, yearly. # TODO: historical
    n = 2 #number of hops used to generate the neighborhoods
    num_nbh = 60000 #number of neighbourhoods to generate
    num_nbh_VT_factor = 0.25 #0.2
    modelrootname = 'DO025_lr0005'

    # set paths and datasetname
    dataset_name = "Berlin_" + datasetrootname
    dataset_version = ' '.join(datasetrootname.split('_')[1:])
    rootpath = os.path.dirname(os.path.realpath(__file__)) #Abs path of AccIndex.git
    data_path = os.path.join(rootpath,'data')
    splitsuf = get_splitsuf(split_strategy)
    graphdatapath = os.path.join(data_path, dataset_name, "Graphs_split"+split_strategy+'_'+splitsuf)

    num_nbh_train = num_nbh*(1-2*num_nbh_VT_factor) if split_strategy=='NBH' else num_nbh
    num_nbh_dict = {"train" : int(num_nbh_train),
                    "val" : int(num_nbh*num_nbh_VT_factor),
                    "test" : int(num_nbh*num_nbh_VT_factor)}
    splitsizestr = ''.join([key+str(int(val)) for key,val in num_nbh_dict.items()])
    NBHdatapath=os.path.join(graphdatapath, 'nbhs_graphs_nhop%d'%(n),splitsizestr)
    model_path = os.path.join(NBHdatapath, 'models')
    model_name = modelrootname + "_20epochs.pt"

    # set path and datasetname
    #dataset_path = os.path.join(data_path, dataset_name)
    #NBHdatapath = os.path.join(dataset_path, 'nbhs_graphs_n%d_numnbh%d'%(n,num_nbh))
    G = pickle.load(open(os.path.join(graphdatapath, "test.pkl"), 'rb'))
    print('G loaded')

    # Density and urban utilisation data downloaded via QGIS from https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s06_06ewdichte2021
    # File can be found on Dropbox
    population_density_data = 'Berlin_pop_density.csv'
    density_df = pd.read_csv( 'data/' + population_density_data)
    density_df['geometry'] = density_df['geometry'].apply(wkt.loads)
    
    # Call the predict function
    start_time = time.time()
    main_predict(G, model_path, model_name, density_df)
    print("Total elapsed time: %s seconds ---" % (time.time() - start_time))