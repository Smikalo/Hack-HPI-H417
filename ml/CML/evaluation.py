import matplotlib.pyplot as plt
import os
import graphviz
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score 
from sklearn.tree import export_graphviz
from xgboost import plot_tree as xgb_plot_tree


def get_skill_scores(D_yyhat):
    scores = {}
    for key in D_yyhat.keys():
        y = D_yyhat[key]['y']
        yhat = D_yyhat[key]['yhat']

        #calculate scores
        R2, MAE, MSE = calc_score(y,yhat)
        scores[key]= {'R2':R2,'MAE':MAE,'MSE':MSE}

        print( f"%s scores: R2 = %.3f MAE:%4.3f, MSE: %4.3f (n:%d)"%(key,R2,MAE,MSE,len(y)))

    return scores


def skill_vis(D_yyhat,scores,Savepath,rootfigname='figure'):
    
    #Scatter plots y vs yhat
    fig, axall = plt.subplots(1,len(D_yyhat), figsize=(4*len(D_yyhat),4))
    plt.subplots_adjust(left=0.1,bottom=0.15,right=0.95,top=0.9,wspace=0.2,hspace=0.4)

    for i,key in enumerate(D_yyhat.keys()):
        y = D_yyhat[key]['y']
        yhat = D_yyhat[key]['yhat']
        scoresK = scores[key]
        if len(D_yyhat)>1:
            ax = axall[i]
        else:
            axall
        ax.plot(y,yhat,'.')
        ax.plot([0,y.max()],[0,y.max()],'-')
        ax.set(xlabel='True # accidents', ylabel='Predicted # accidents')
        scorestr ='n: %d\nR2: %5.2f\nMAE: %3.2f\nMSE: %3.2f'%(len(y),scoresK['R2'],scoresK['MAE'],scoresK['MSE'])
        ax.text(0.05,0.78, scorestr, transform=ax.transAxes)
        ax.set_title(key)
        ax.set_xlim([0,18])
        ax.set_ylim([0,18])
        ax.set_xticks([0,5,10,15])
        ax.set_yticks([0,5,10,15])
        figpath = os.path.join(Savepath, rootfigname + '_Scatter.png')
        fig.savefig(figpath)

    print(f'Saved figure: {figpath}')
    
    
def calc_score(y,yhat):
    R2 = r2_score(y, yhat)
    MAE = mean_absolute_error(y,yhat)
    MSE = mean_squared_error(y,yhat,squared=True)
    #RMSE = mean_squared_error(y,yhat,squared=False)
    
    return (R2,MAE,MSE)


def model_vis(options,model,Savepath,rootfigname): #,features
    if options['model'] == 'DTR':
        treereg_graphviz(model,Savepath,rootfigname) #,features
    elif options['model'] == 'XGBR':
        xgbreg_graphviz(model,Savepath,rootfigname)
    return


def xgbreg_graphviz(model,Savepath,rootfigname): #,features
    figpath = os.path.join(Savepath, rootfigname + '_XGBR_graphviz.png')
    fig, ax = plt.subplots(1,1, figsize=(30,10))
    xgb_plot_tree(model, num_trees=model.best_iteration, ax=ax) #,feature_names=features
    fig.savefig(figpath)
    print(f'Saved figure: {figpath}')


def treereg_graphviz(model,Savepath,rootfigname): #features
    figpath = os.path.join(Savepath, rootfigname + '_DTR_graphviz')
    # DOT data
    dot_data = export_graphviz(model, out_file=None, 
                                #feature_names=features,  
                                filled=True)
    # Draw graph
    graph = graphviz.Source(dot_data, format="png") 
    graph.render(figpath)
    print(f'Saved figure: {figpath}')
