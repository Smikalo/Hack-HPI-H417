#libraries
import networkx as nx
import os
import numpy as np
import pandas as pd
import pickle
import sys
# local functions
#Abs path of AccIndex.git
rootpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(rootpath, 'utils'))
from utility_funcs import get_splitsuf

def main(test_mode=False):
    #get graph data
    data_path = os.path.join(rootpath,'data')
    regioncode = 'DSR' if test_mode else 'Berlin'
    datasetrootname = "v221217_NodeR0_Full_NoEnc"
    split_strategy = 'yearly' #options: randomsamp, yearly, historical

    # set paths and datasetname
    dataset_name = regioncode + "_" + datasetrootname
    data_path = os.path.join(rootpath,'data')
    splitsuf = get_splitsuf(split_strategy)
    Gdatapath = os.path.join(data_path, dataset_name, "Graphs_split"+split_strategy+'_'+splitsuf)
    CSVdatapath = os.path.join(Gdatapath, 'csv')
    os.mkdir(CSVdatapath) if not os.path.exists(CSVdatapath) else print('Overwriting contents in existing csv dir')

    for split in ['train','val','test']:
        f_pkl = os.path.join(Gdatapath, split+'.pkl')
        f_csv_nodes = os.path.join(CSVdatapath, split + '_nodes.csv')
        f_csv_edges = os.path.join(CSVdatapath, split + '_edges.csv')

        Gfull_raw = get_data(f_pkl)

        Gfull, nodenum, edgenum, node_attr_final, edge_attr_final = prepare_Gfull(Gfull_raw)

        NodesDF = convert_Gnodes2DF(Gfull, nodenum, node_attr_final)
        EdgesDF = convert_Gedges2DF(Gfull, edgenum, edge_attr_final)

        DF_D = {'N': NodesDF, 'E':EdgesDF}
        #DF_D = encode_edgefeats(DF_D)
        DF_D = convert_dtypes(DF_D)
        #DF_D = handle_nans(DF_D)
        #DF_D = remove_outliers(DF_D)

        DF_D['N'].to_csv(f_csv_nodes)
        DF_D['E'].to_csv(f_csv_edges)
        print(f'created files:\n{f_csv_nodes}\n{f_csv_edges}')
    return

#only for float (seems like there is no need)
def remove_outliers(DF_D): 
    for DFname, DF in DF_D.items():
        print(f'Eliminating outliers from {DFname} features')
        for col in DF.columns:
            #remove outliers
            #print('dtype(%s): %s'%(col, DF[col].dtype))
            if DF[col].dtype == 'Float64':
                print(f' - outliers in {col}:',end='')
                # calculate summary statistics
                data_mean, data_std = np.mean(DF[col]), np.std(DF[col])
                # identify outliers
                cut_off = data_std * 6
                lower, upper = data_mean - cut_off, data_mean + cut_off
                # identify outliers
                nani = DF[col]>upper
                if nani.sum()>0:
                    print([x for x in DF.loc[nani==True, col]])
                else:
                    print(' none.')
                DF.loc[nani] = np.nan
    return(DF_D)


def encode_edgefeats(DF_D):
    Binarycols = ['reversed', 'oneway', 'bridge']
    for DFname,DF in DF_D.items():
        print(f'Encoding: {DFname} features')
        for col in Binarycols:
            if col in DF.columns:
                print(f' - binary encoding: {col}')
                DF[col].replace(True, 1, inplace=True)
                DF[col].replace(False,0, inplace=True)
                DF[col].replace('yes', 1, inplace=True)
                DF[col].replace('no',0, inplace=True)
    return(DF_D)


def handle_nans(DF_D):
    nancols = ['']
    for DFname, DF in DF_D.items():
        print(f'Handling nans from: {DFname} features')
        for col in nancols:
            if col in DF.columns:
                print(f' - removing nans: {col}')
    return(DF_D)


def convert_dtypes(DF_D):
    Str2Floatcols = ['width', 'maxspeed']  #'length', 'x', 'y']
    for DFname,DF in DF_D.items():
        print(f'Converting: {DFname} features from str to float')
        for col in Str2Floatcols:
            if col in DF.columns:
                DF[col] = DF[col].astype(str)
                if col == 'width':
                    DF[col] = DF[col].str.replace(' m', '')
                if col == 'maxspeed':
                    DF[col] = DF[col].str.replace('DE:urban', '50')
                    DF[col] = DF[col].str.replace('walk', '5')
                DF[col] = DF[col].replace('none', None)
                DF[col] = DF[col].replace('None', None)
                DF[col] = DF[col].replace('<NA>', None)
                if DF[col].dtype != 'Float64':
                    print(f' - converting {col}')
                    DF[col] = DF[col].astype(dtype='Float64')
                    DF[col] = DF[col].replace({pd.NA: np.nan})
    return(DF_D)
        

def convert_Gnodes2DF(Gfull, nodenum, colnames_full):
    #this is custom-ordered list of columns
    colnames=['index', 'nr_accidents', 'x', 'y', 'street_count', 'highway']#, 'ref']
    #check if everything is there
    if set(colnames) < set(colnames_full):
        raise(Exception('missing feature in colnames:%s'%list(set(colnames_full)-set(colnames))))
    DFnodes = pd.DataFrame(data = None, columns=colnames, index=list(range(nodenum)))
    nodeno=-1
    for node_i, n in Gfull.nodes(data=True):
        nodeno += 1
        #print(nodeno)
        for key in colnames:
            #print(key, end=', ')
            if key == 'index':
                value = node_i
            elif key in n.keys():
                if type(n[key]) == list:
                    #replace the empty lists with something more informative
                    if len(n[key])==0:
                        if key == 'highway':
                            value = 'None'
                        else:
                            raise(Exception(f'(Node) {key}: is empty list, undefined action'))
                    else:
                        #replace the lists with their average or join
                        #allfloat = all([isinstance(el, float) for el in n[key]])
                        try:
                            values = [float(el) for el in n[key]]
                            if key  == 'nr_accidents':
                                value = np.sum(values)
                                print(f'Node-{node_i}: summing nr_accidents')
                            else:
                                value =  np.mean(values)
                        except:
                            value = ','.join(n[key])
                else:
                    try:
                        value = float(n[key])
                    except:
                        value = n[key]
            DFnodes[key].loc[nodeno] = value
    return DFnodes

def convert_Gedges2DF(Gfull, edgenum, colnames_full):
    #this is custom-ordered list of columns
    colnames = ['Sindex', 'Rindex', 'name', 'nr_accidents', 'length','width','lanes','maxspeed',
                'reversed','oneway','junction','access','bridge','service','tunnel','highway',
                ] #,'area','ref']
    #check if everything is there
    if set(colnames) < set(colnames_full):
        raise(Exception('missing feature in colnames:%s'%list(set(colnames_full)-set(colnames))))
    DFedges = pd.DataFrame(data = None, columns=colnames, index=list(range(edgenum)))
    edgeno=-1
    for s,r,e in Gfull.edges(data=True):
        edgeno += 1
        #print(edgeno)
        for key in colnames:
            #print(key, end=', ')
            if key == 'Sindex':
                value = s
            elif key == 'Rindex':
                value = r
            elif key in e.keys():
                if type(e[key]) == list:
                    #replace the empty lists with something more informative
                    if len(e[key])==0:
                        if key =='name':
                            value = 'unknown'
                        elif key in ['reversed', 'oneway']: 
                            value = 'None'  #False
                        elif key in ['bridge']:
                            value = 'None' #'no'
                        elif key in ['width','lanes']:
                            value = pd.NA
                        elif key in ['maxspeed', 'junction', 'service', 'tunnel', 'highway', 'access', 'tunnel']:
                            value = 'None'
                        else:
                            raise(Exception(f'(Edge) {key}: is empty list, undefined action'))
                    else:
                        #replace the lists with their average or join
                        #allfloat = all([isinstance(el, float) for val in e[key]])
                        if key in ['oneway','reversed']: #we don't want to convert True/False to 0/1
                            values = [str(el) for el in e[key]]
                            value = ','.join(values)
                        else:
                            try:
                                values = [float(el) for el in e[key]]
                                if key  == 'nr_accidents':
                                    value = np.sum(values)
                                    print(f'Edge-{edgeno}: summing nr_accidents')
                                else:
                                    value =  np.mean(values)
                            except:
                                value = ','.join(e[key])
                        #raise(Exception(f'(Edge) {key}: is non-empty list, undefined action'))
                else:
                    if key in ['oneway','reversed']: #we don't want to convert True/False to 0/1
                            value = str(e[key])
                    else:
                        try:
                            value = float(e[key])
                        except:
                            value = e[key]
            DFedges[key].loc[edgeno] = value
    return DFedges


def prepare_Gfull(Gfull):
    #attributes contained
    #edge_attr = ['width' ,'tunnel', 'service', 'reversed', 'v_original', 'bridge', 
    #            'maxspeed', 'name', 'access', 'junction', 'highway', 'oneway', 
    #            'u_original', 'geometry', 'accidents_list','width',  'lanes']
    #node_attr = ['accidents_list', 'osmid', 'highway', 'lat', 'lon']

    #attributes to be removed (eg, duplicates, second order information, etc)
    edge_attr2pop = ['accidents_list', 'osmid', 'geometry', 'u_original', 'v_original', 'ref', 'area'] 
    node_attr2pop = ['accidents_list', 'osmid', 'osmid_original', 'lat', 'lon', 'ref']

    #purge unwanted fields
    nodeno=0
    unique_attrs = set([])
    for i, n in Gfull.nodes(data=True):
        nodeno += 1
        #n["nr_acc"] = len(n['accidents_list']) #this was done earlier
        for k in node_attr2pop:
            n.pop(k, None)
        unique_attrs = unique_attrs | set(n.keys())
    node_attr_final  = ['index'] + list(unique_attrs)

    edgeno=0
    unique_attrs = set([])
    for s, r, e in Gfull.edges(data=True):
        edgeno += 1
        #e["nr_acc"] = len(e['accidents_list'])
        for k in edge_attr2pop:
            e.pop(k,None)
        unique_attrs = unique_attrs | set(e.keys())
    edge_attr_final  = ['Sindex', 'Rindex'] + list(unique_attrs)

    return Gfull, nodeno, edgeno, node_attr_final, edge_attr_final

def get_data(fname):
    with open(fname, 'rb') as f:
        Gfull = pickle.load(f)
        return(Gfull)


if __name__=='__main__':
    main()