#libraries
import os
import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import spearmanr,kruskal,chi2_contingency
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import sys
# local functions
rootpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(rootpath, 'utils'))
from utility_funcs import get_splitsuf

def main_analyze(test_mode=False, vis_featfreqs=True, vis_featcorrs=True):
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
    Plotdatapath = os.path.join(Gdatapath, 'plots')
    os.mkdir(Plotdatapath) if not os.path.exists(Plotdatapath) else print('Overwriting contents in existing plot dir')

    for split in ['train','val','test']:
        #DF_D: dataframe dictionary
        DF_D = get_csv_data(CSVdatapath, split)

        #histogram (for float vars) and barplots (for categorical variables)
        if vis_featfreqs:
            feat_freqs(DF_D,
            Plotdatapath,
            split
            )

        if vis_featcorrs:
            feat_corrs(DF_D,
            Plotdatapath,
            split,
            cat_rel='CT-chi2' #options: CT-chi2, box-KW
            )

    print('All tasks completed.')
    return


def feat_corrs(DF_D, rootfolder, rootfname, cat_rel):

    labelcol = 'nr_accidents'
    cols2plot_D = {
        'N': ['nr_accidents', 'x', 'y', 'street_count', 'highway'], #, 'ref'
        'E': ['nr_accidents', 'length', 'maxspeed', 'width', 'lanes', 
              'highway', 'oneway','bridge','reversed','junction',
              'tunnel','service', 'access'] #,'ref', 'area']
            }

    spcols = 5
    sprows = math.ceil(len(cols2plot_D['N'] + cols2plot_D['E'])/spcols)
    fig = plt.figure(figsize=(spcols*4, sprows*4.5))
    plt.subplots_adjust(bottom=0.07, left=0.05, top = 0.95, right=0.95, hspace=0.5, wspace=0.25)

    i=0
    for datasetname, DF in DF_D.items():
        if datasetname == 'N':
            cols2plot = cols2plot_D['N']
            DF = DF_D['N']
            print('Plotting Node attribute corrs:')
        elif datasetname == 'E':
            cols2plot = cols2plot_D['E']
            DF = DF_D['E']
            print('Plotting Edge attribute corrs:')
        for col in cols2plot:
            i += 1
            ax = plt.subplot(sprows, spcols, i)
            if DF[col].dtype=='Float64':
                allnan = np.isnan(DF[col]).all()
            else:
                allnan = pd.isnull(DF[col]).all()
            if (col not in DF.columns) or allnan:
                ax.set_title('(%s) %s: not available'%(datasetname,col))
                continue
            titlestr = r'n: %s'%(DF[col].notnull().sum())
            if col == 'nr_accidents':
                DF[col].plot.hist(ax=ax, bins=10, log=True)
                statstr = ''
            elif DF[col].dtype == 'float64':
                print(' %s) dtype(%s): %s scatter'%(i, col, DF[col].dtype))
                logscale = True if col in ['length'] else False
                DF.plot.scatter(x=col, y=labelcol, ax=ax, logx=logscale, logy=False)
                ax.set_ylabel(f"({datasetname}) {labelcol}")
                coef, p = spearmanr(DF[col], DF[labelcol], nan_policy='omit')
                statstr = r'$\rho$ = %3.2f $p$ = %5.4f'%(coef,p)
                #ax.text(0.03,0.95,statstr,transform = ax.transAxes)
            else: #['int64','object']
                if cat_rel == 'box-KW':
                    print(' %s) dtype(%s): %s box-KW'%(i, col, DF[col].dtype))
                    sns.boxplot(data=DF, x=col, y=labelcol)
                    ax.set_ylabel("# accidents")
                    if DF[col].dtype == 'object':
                        le = LabelEncoder()
                        series_col = le.fit_transform(DF[col])
                    else:
                        series_col = DF[col]
                    H, p = kruskal(series_col, DF[labelcol], nan_policy='omit')
                    statstr = r'$H$ = %3.2f $p$ = %5.4f'%(H,p)
                elif cat_rel == 'CT-chi2':
                    print(' %s) dtype(%s): %s CT-chi2'%(i, col, DF[col].dtype))
                    # Generate the contingency table
                    contingency_table = pd.crosstab(DF[labelcol],DF[col])
                    sns.heatmap(contingency_table, cmap="YlGnBu") #, annot=True, fmt='.2f', vmin=0.0, vmax=100.0)
                    #mosaic(DF, [col, labelcol], ax=ax, label_rotation=[30,0])
                    # Compute the chi-squared statistic and p-value
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    statstr = r"${\chi}^2$ = %.3f, $p$ = %.3f"%(chi2,p)
                #ax.text(0.03,0.95,statstr,transform = ax.transAxes)
                plt.xticks(rotation=45)
                ax.set_ylabel(f"({datasetname}) {labelcol}")
            ax.set_xlabel(f"({datasetname}) {col}")    
            ax.set_title(titlestr + ' ' + statstr)
    figfname = os.path.join(rootfolder, rootfname+'_feat_corrs.png')
    fig.savefig(figfname)
    print(f'figure saved: {figfname}')
    return


def feat_freqs(DF_D, rootfolder, rootfname):
    cols2plot_D = {
        'N': ['nr_accidents', 'x', 'y', 'street_count', 'highway'], #, 'ref'
        'E': ['nr_accidents', 'length', 'maxspeed', 'width', 'lanes', 'highway',
            'oneway','bridge','reversed','junction','tunnel','service',
            'access'] #,'ref', 'area']
            }

    spcols = 5
    sprows = math.ceil(len(cols2plot_D['N'] + cols2plot_D['E'])/spcols)
    fig = plt.figure(figsize=(spcols*4, sprows*4.5))
    plt.subplots_adjust(bottom=0.07, left=0.05, top = 0.95, right=0.95, hspace=0.5, wspace=0.25)

    i=0
    for datasetname, DF in DF_D.items():
        if datasetname == 'N':
            cols2plot = cols2plot_D['N']
            DF = DF_D['N']
            print('Plotting Node attribute frequencies:')
        elif datasetname == 'E':
            cols2plot = cols2plot_D['E']
            DF = DF_D['E']
            print('Plotting Edge attribute frequencies:')
        for col in cols2plot:
            i += 1
            ax = plt.subplot(sprows, spcols, i)
            if col not in DF.columns:
                plt.title('(%s) %s\nNot available'%(datasetname,col))
                continue
            title = '(%s) %s\n# non-null: %s'%(datasetname,col,DF[col].notnull().sum())
            if DF[col].dtype == 'float64' or col=='nr_accidents':
                print(' %s) dtype(%s): %s hist'%(i, col, DF[col].dtype))
                logscale = True if col in ['length', 'nr_accidents'] else False
                DF[col].plot.hist(ax=ax, bins=10, log=logscale)
                ax.set_ylabel("")
            else: #['int64','object']
                print(' %s) dtype(%s): %s bar'%(i, col, DF[col].dtype))
                DF[col].value_counts().plot.bar(ax=ax, rot=45)
            ax.set_title(title)
    figfname = os.path.join(rootfolder, rootfname+'_feat_freqs.png')
    fig.savefig(figfname)
    print(f'figure saved: {figfname}')
    return


def get_csv_data(rootfolder,rootfname):
    f_csv_nodes = os.path.join(rootfolder, rootfname + '_nodes.csv')
    f_csv_edges = os.path.join(rootfolder, rootfname + '_edges.csv')

    with open(f_csv_nodes) as f:
        DFnodes = pd.read_csv(f)
    with open(f_csv_edges) as f:
        DFedges = pd.read_csv(f) #, dtype={'reversed':'Int64', 'oneway':'Int64', 'bridge':'Int64'})

    return {'N': DFnodes, 'E':DFedges}


if __name__=='__main__':
    main_analyze()