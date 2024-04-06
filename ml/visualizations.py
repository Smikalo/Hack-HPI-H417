# import packages
from osmnx import plot_graph, plot_graph_routes, plot_graph_route
#from taxicab.plot import plot_graph_route as tc_plot_graph_route
import matplotlib.pyplot as plt
import streamlit as st

def visualize_training_hist(ModelSkillDict, modelskill_path, expname):
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    colors = ['b','g','r']
    lstyle = ['-', ':', None]
    i=0
    numepochs=None
    for key,scores in ModelSkillDict.items():
        if isinstance(scores, list):
            numepochs = len(scores)
            plt.plot(range(numepochs), scores, color=colors[i], linestyle = lstyle[i], label=key)
        else:
            continue
            #plt.plot(numepochs-1,scores, color=colors[i], linestyle =lstyle[i], marker='o', label=key)
        i += 1
    ax.set_xlim([0,numepochs-1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (#Acc/Year)')
    ax.set_title(expname)
    plt.legend()

    figpath = modelskill_path.replace('.pkl','')
    fig.savefig(figpath)
    print(f'Figure saved:{figpath}')

def plot_map_box(G, routes, c1, c2, parameters):
    st.write(" Generating map")

    if len(routes) == 1:

        accident_count = 0
        for node_i in range(len(routes[0])-1):
            edge_data = G.get_edge_data(routes[0][node_i],routes[0][node_i+1])

            if not 'pred_acc' in edge_data[0]:
                continue
            else:
                pred_acc = edge_data[0]['pred_acc']
                if len(pred_acc) == 0:
                    acc_count_i = 0
                else:
                    #print('print node: %d, pred. acc: %s'%(node_i, str(edge_data[0]['pred_acc'])))
                    acc_count_i  = float(pred_acc)
                accident_count += acc_count_i
        
        #print('cumulative acc count: %4.2f'%accident_count)
        fig, ax = plot_graph_route(G,
                                    routes[0],
                                    route_color='b',
                                    route_linewidth=4,
                                    node_size=0,
                                    show=False,
                                    close=False,
                                    figsize=(10, 10))
        
        ax.scatter(c1.longitude,
                   c1.latitude,
                   c='yellow',
                   s=200,
                   label='orig',
                   marker='o')
        ax.scatter(c2.longitude,
                   c2.latitude,
                   c='yellow',
                   s=200,
                   label='dest',
                   marker='o')
        ax.set_title("Estimated cumulative # accidents/year on the route: %4.2f" %accident_count,
                     fontsize=20,
                     c="red")
        figname = 'plot_map_route.png'
    else:
        fig, ax = plot_graph(G,
                             routes,
                             node_size=5,
                             show=False,
                             close=False,
                             figsize=(10, 10))
        figname = 'plot_map.png'

    padding = 0.005
    ax.set_ylim([
        min([c1.latitude, c2.latitude]) - padding,
        max([c1.latitude, c2.latitude]) + padding
    ])
    ax.set_xlim([
        min([c1.longitude, c2.longitude]) - padding,
        max([c1.longitude, c2.longitude]) + padding
    ])
    plt.savefig(figname)
    plt.close()
