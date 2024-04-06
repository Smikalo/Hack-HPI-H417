"""
Main module of the AccInd using Classical Machine Learning approaches

Author:
Onur Kerimoglu (kerimoglu.o@gmail.com)
"""

# import packages
import os
import pickle

# import functions from project modules
from utils.utility_funcs import get_splitsuf
from CML.rawdata import get_data
from CML.models import construct_model
from CML.pipelines import get_pipelines
from CML.FPE import fit_predict_evaluate
from CML.utilities import get_expname_datetime


def main(test_mode=False):
    """
    Args: None
    Returns: None
    """
    #General parameters
    options = {
    "features_n": ['street_count','x','y','highway'],
    "features_e": ['length','maxspeed','highway','oneway','bridge','reversed','junction','tunnel','service','access'],
    "model": "XGBR", #DTR, XGBR
    "name": "ALLfeats",
    "mode": 'test', #'test' 'gridsearch'
    "do_nodes": False,
    "do_edges": True,
    }
    rootpath = os.path.dirname(os.path.realpath(__file__)) #Abs path of AccIndex.git
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
    # construct an experiment name based on current date time
    expname = get_expname_datetime(options)
    Respath = os.path.join(Gdatapath, 'CML', expname)
    os.makedirs(Respath) if not os.path.exists(Respath) else print('Overwriting contents in existing plot dir')

    # Load the data
    DF_N_train, DF_E_train, DF_N_val, DF_E_val, DF_N_test, DF_E_test = get_data(CSVdatapath)

    # Build the model for N
    model_N,model_E,GSparameters_N,GSparameters_E = construct_model(opt=options["model"])
    
    # Get the pipeline for the node and edge models
    pl_N, pl_E = get_pipelines(options, DF_N_train, DF_E_train, model_N, model_E)

    if options['do_nodes']:
        print('\nTraining the model for Nodes:')
        # do the fitting and predictions for nodes
        scores_N = fit_predict_evaluate(
            options,
            pl_N,
            DF_N_train,
            DF_N_test,
            DF_N_val,
            GSparameters_N,
            expname=expname+'_N',
            Savepath=Respath
        )
        with open(os.path.join(Respath, expname+"_N_scores.pkl"), "wb") as handle:
            pickle.dump(scores_N, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(Respath, expname+"_N_pipeline.pkl"), "wb") as handle:
            pickle.dump(pl_N, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if options['do_edges']:
        print('\nTraining the model for Edges:')
        # do the fitting and predictions for nodes
        scores_E = fit_predict_evaluate(
            options,
            pl_E,
            DF_E_train,
            DF_E_test,
            DF_E_val,
            GSparameters_E,
            expname=expname+'_E',
            Savepath=Respath
        )
        with open(os.path.join(Respath, expname+"_E_scores.pkl"), "wb") as handle:
            pickle.dump(scores_E, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(Respath, expname+"_E_pipeline.pkl"), "wb") as handle:
            pickle.dump(pl_E, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # pickle options, scores and pipelines
    with open(os.path.join(Respath, expname+"_options.pkl"), "wb") as handle:
        pickle.dump(options, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved the options, scores and pipeline in: {Respath}")
    return

if __name__ == "__main__":
     main()