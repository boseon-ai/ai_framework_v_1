import os
import numpy as np
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

def directory_check(dir_path):
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

def plot_loss_graph(t_loss, v_loss, model_name, plot_dir='./loss_graphs/'):
    directory_check(plot_dir)
    fig = os.path.join(plot_dir, model_name+'.png')
    plt.figure(figsize=[14,5])
    plt.title(model_name)
    plt.plot(t_loss, label='train loss')
    plt.plot(v_loss, label='valid loss')
    plt.legend()
    plt.grid()
    plt.savefig(fig)
    return fig

def save_rst_scalable(dict_pred_label, rst_dir='./scalable_plots/'):
    # dict_pred_label contains dataframes for prediction results and label
    if os.path.exists(rst_dir) == False:
        os.makedirs(rst_dir)
    
    labels = dict_pred_label['labels']
    models = sorted(dict_pred_label.keys()) 
    
    xticks = np.arange(labels.shape[0])
    
    for key in labels.keys():
        data = []
        layout = dict(title=str(key), xaxis=dict(rangeslider=dict(visible=True)))
        for model in models:
            data.append(go.Scatter(x=xticks, y=dict_pred_label[model][key], name=model))
            
        fig = dict(data=data, layout=layout)
        plotly.offline.plot(fig, filename='{}.html'.format(rst_dir + str(key)))
        
    print ('plots are saved in {}.'.format(rst_dir))
