import os, sys
import osmnx as ox
import pickle
import pandas as pd
import plotly.express as px
import numpy as np


def main(mapbox_access_token, plotmap=False, plotscatter=True, create_pred_graph_toggle=False, NAIVEPRED=False):

    rootpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) #Abs path of AccIndex.git
    data_path = os.path.join(rootpath,'data')
    pred_fpath = os.path.join(data_path, 'Berlin_v221217_NodeR0_nhop2/Graphs_splityearly_T1821_V1920_T21/nbhs_graphs_nhop2/train60000val15000test15000/models/predictions')
    graphdatapath = 'data/Berlin_v221217_NodeR0_nhop2/Graphs_splityearly_T1821_V1920_T21'
    
    rawgraphdatafname = os.path.join(data_path, pred_fpath, "G_raw.pkl")
    accproc_fpath = os.path.join(data_path, 'Berlin_v221217_NodeR0_Full_NoEnc/Graphs_splityearly_T1819_V20_T21/csv/')

    pred_fname = os.path.join(pred_fpath, 'global_edge_preds.csv')
    #pred_fname = "predictions_restored.csv"
    accraw_fname = os.path.join(data_path, 'acc_berlin_2018_to_2022_clean.csv')
    #accraw_fname = "../data/acc_berlin_2018_to_2022_clean.csv"

    fnamedict= {'Train': 'train.pkl', 'Val': 'val.pkl'}

	# datasetname prefix and suffixes

    if NAIVEPRED and (not plotmap) and (not create_pred_graph_toggle):
        #naive predictions are only intented to be used for scatterplotting
        preds = get_preds_naive(os.path.join(graphdatapath, fnamedict['Train']))
    else:
        preds = get_preds(pred_fname)

    if plotmap:
        plot_map (preds, accraw_fname, mapbox_access_token)

    if plotscatter:
        #plot_scatter_from_csvG(rootpath, preds, accproc_fpath, fnamedict, pred_fpath, 'Globalpreds_vs_obs')
        plot_scatter_from_obsG(rootpath, graphdatapath, preds, fnamedict, pred_fpath, NAIVEPRED)

    if create_pred_graph_toggle:
        create_pred_graph(rawgraphdatafname, pred_fname, preds)


def plot_scatter_from_obsG(rootpath, graphdatapath, preds, fnamedict, pred_fpath, NAIVEPRED = False):
    sys.path.append(os.path.join(rootpath, 'CML'))
    from evaluation import skill_vis, get_skill_scores

    #For edges
    D_yyhat = {}
    for split,fname in fnamedict.items():
        
        y,yhat = get_y_yhat(os.path.join(graphdatapath,fname),preds)

        D_yyhat[split] = {'y':y, 'yhat':yhat}

    scores = get_skill_scores(D_yyhat)
    if NAIVEPRED:
        fname = 'obs_vs_preds_naive'
    else:
        fname = 'obs_vs_preds'
    skill_vis(D_yyhat,scores,pred_fpath,fname)


def get_y_yhat(graph_fname, preds):
    DFpreds = preds[['start_node','end_node','predicted_edge_acc']]
    DFpreds = DFpreds.rename(columns={'predicted_edge_acc':'acc'})
    DFpreds = DFpreds.drop_duplicates()

    G = pickle.load(open(graph_fname,'rb'))
    DFobs = pd.DataFrame({'start_node': [e[0] for e in G.edges(data=True)],
                          'end_node': [e[1] for e in G.edges(data=True)],
                          'acc' : [e[2]['nr_accidents'] for e in G.edges(data=True)]
                          })
    DFobs = DFobs.drop_duplicates()
    
    #to make sure that the edges match
    DFmerged = pd.merge(DFpreds, DFobs, on=['start_node','end_node'], how='left', suffixes=('_pred','_obs'))

    return DFmerged['acc_obs'].to_numpy(),DFmerged['acc_pred'].to_numpy()


def plot_scatter_from_obscsv(rootpath, preds,accproc_fpath,fnamedict, Savepath, expnameroot):
    sys.path.append(os.path.join(rootpath, 'CML'))
    from evaluation import skill_vis, get_skill_scores

    #For edges
    D_yyhat = {}
    for split,fname in fnamedict.items():
        accproc_fname = os.path.join(accproc_fpath, fname)
        print(f'Reading obs file:{accproc_fname}')
        obs = pd.read_csv(accproc_fname)

        y, yhat = match_preds_obs_edges(preds, 
                                        obs, 
                                        predcol='predicted_edge_acc', 
                                        obscol='nr_accidents', 
                                        mergecols=['start_node', 'end_node'])

        D_yyhat[split] = {'y':y, 'yhat':yhat}

    scores = get_skill_scores(D_yyhat)
    skill_vis(D_yyhat,scores,Savepath,expnameroot+'_edges')


def match_preds_obs_edges(DFp,DFo,predcol,obscol,mergecols):
    
    DFpo_raw = pd.merge(DFp, DFo, on=mergecols, how='left', suffixes=('_pred','_obs'))
    #DFpoy_raw = DFpo_raw[[predcol,obscol]]
    #DFpoy = DFpoy.dropna(axis=0, how='any')
    DFpo = DFpo_raw.dropna(axis=0, subset = [predcol,obscol], how='any')
    yhat = DFpo[predcol]
    y = DFpo[obscol]

    return y,yhat


def create_pred_graph(rawgraphdatafname, pred_fname, pred_df):
    print('Creating G with predicted accidents')
    if os.path.exists(rawgraphdatafname):
        G = pickle.load(open(rawgraphdatafname, 'rb'))
    else:
        G = get_G('Berlin')

    acc_mean = pred_df['predicted_edge_acc'].mean()

    for e_i,e in enumerate(G.edges(data=True)):
        row_match = (pred_df['start_node']==e[0])&(pred_df['end_node']==e[1])
        if not any(row_match): # actually this should not happen
            print(f' Graph edge {e_i} (start node: {e[0]}, end node:{e[1]}) not found in predictions. will assign mean value')
            e[2]["pred_acc"] = acc_mean #but if happens, fill it with a mean value
        else:
            e[2]["pred_acc"] = pred_df.iloc[np.where(row_match)[0][0]]['predicted_edge_acc']
    
    #Berlin_noder0_v221216_Graphs_splityearly_T1819_V20_T21_test_withpred_acc.pkl
    predG_fname = pred_fname.replace('.csv', '_G.pkl')
    pickle.dump(G, file=open(predG_fname,'wb'))
    print(f'Pickled:{predG_fname}')


def get_G(city_name, node_tolerance=10):
    print(f'Generating Graph object for:{city_name} with node tolerance: {node_tolerance}')
    G = ox.graph_from_place(city_name, network_type="drive")
    G = ox.project_graph(G)
    print("consolidate intersections")
    G = ox.consolidate_intersections(G, rebuild_graph=True, tolerance=node_tolerance, dead_ends=False)
    # project back to geo coordinate system
    G = ox.project_graph(G, to_crs="EPSG:4326")

    return G


def plot_map(df, accraw_fname, mapbox_access_token):

    if mapbox_access_token == None:
        raise(Exception('At least one argument is required when calling this script: mapbox_access_token'))

    accraw = pd.read_csv(accraw_fname)

    px.set_mapbox_access_token(mapbox_access_token)

    # Alternatively enter predicted_edge_acc as color
    fig = px.density_mapbox(df,lat='middle_lat', lon='middle_lon',  radius =20, z="predicted_edge_acc",
                            color_continuous_scale=px.colors.sequential.Reds,  zoom=10)

    fig.add_trace(px.scatter_mapbox(accraw,lat="lat",lon="lon", opacity=0.5).data[0])

    fig.show()


def get_preds_naive(graph_fname):
    #only intended to be used for scatterplotting
    G = pickle.load(open(graph_fname,'rb'))
    DFpred = pd.DataFrame({'start_node': [e[0] for e in G.edges(data=True)],
                           'end_node': [e[1] for e in G.edges(data=True)],
                           'predicted_edge_acc' : [e[2]['nr_accidents'] for e in G.edges(data=True)]
                         })
    return DFpred


def get_preds(predfname):
    df = pd.read_csv(predfname)

    df["sqrt_predicted_edge_acc"] = df["predicted_edge_acc"] - df["predicted_edge_acc"].min() + 0.00001
    df["sqrt_predicted_edge_acc"] = np.sqrt(df["sqrt_predicted_edge_acc"])

    df['middle_lon'] = (df["start_lon"] + df["end_lon"])/2
    df['middle_lat'] = (df["start_lat"] + df["end_lat"])/2

    start_end_pairs = [df['edge'][i].replace('(','').replace(')','').split(',') for i in range(len(df))]
    df['start_node'] =  [int(x[0]) for x in start_end_pairs]
    df['end_node'] =   [int(x[1]) for x in start_end_pairs]

    return df


if __name__ == '__main__':
    program_name = sys.argv[0]
    arguments = sys.argv[1:]
    if len(arguments)<1:
        mapbox_access_token = None
    else:
        #mapbox_access_token = arguments[0]
        mapbox_access_token = ''
    main(mapbox_access_token)
