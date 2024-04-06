import os
import pandas as pd
"""
This module contains functions that deal with raw data.
"""

def get_data(CSVdatapath):
    DF_N_train, DF_E_train = get_csv_data(CSVdatapath, 'train')
    DF_N_val, DF_E_val = get_csv_data(CSVdatapath, 'val')
    DF_N_test, DF_E_test = get_csv_data(CSVdatapath, 'test')
    return DF_N_train, DF_E_train, DF_N_val, DF_E_val, DF_N_test, DF_E_test

def get_csv_data(rootfolder,rootfname):
    f_csv_nodes = os.path.join(rootfolder, rootfname + '_nodes.csv')
    f_csv_edges = os.path.join(rootfolder, rootfname + '_edges.csv')

    with open(f_csv_nodes) as f:
        DFnodes = pd.read_csv(f)
    with open(f_csv_edges) as f:
        DFedges = pd.read_csv(f) #, dtype={'reversed':'Int64', 'oneway':'Int64', 'bridge':'Int64'})

    return DFnodes, DFedges