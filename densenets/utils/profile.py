import torch
import torchvision.models as models
# 
import plotly
import plotly.graph_objs as go
import plotly.express as px
# 
import pandas as pd
import numpy as np
# 
import re
from collections import OrderedDict 
from utils.layout import *

# Convert table to CSV
def table_to_pandas(table):
    
    """
    INPUT -- The table produced by pytorch profiler
    OUTPUT -- Pandas dataframe corresponding to the table
    """
    
    re.sub('-+','',table)
    
    lines = table.split('\n')
    
    del lines[0]
    del lines[1]
    del lines[-1]
    
    for i in range(0,len(lines)):
        lines[i] = re.sub('  +', ',', lines[i])
        lines[i] = re.sub('us', '', lines[i])
        lines[i] = re.sub('ms', '', lines[i])
    
    data_frame = pd.DataFrame([l.split(',') for l in lines[1:]], columns=[x for x in lines[0].split(',')])
    return data_frame


# Layer-by-layer profiling
def profile(model=None, input_size=(1,3,32,32), num_classes=10, reps=1, use_cuda=True):

    """
    INPUT -- A model and the input size to run the model on
    OUTPUT -- A table with layer-by-layer breakdown of function calls in the model
    """


    x = torch.randn(input_size, requires_grad=True)
    target = torch.randn([input_size[0],num_classes]).type(torch.LongTensor)
    target = torch.max(target, 1)[1]

    # If using CUDA
    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        x.cuda()
        target.cuda()
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
   
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(reps):
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return prof.key_averages().table(sort_by='cpu_time')
    
def run_and_visualize_profile_attr(models, input_size=(128,3,32,32), attr='CPU time'):

    """
    INPUT -- Models -- A dictionary, key: name, value: model
    OUTPUT -- A graph
    """

    fig = go.Figure()

    for name in models.keys():
        model = models[name]
        data_frame = table_to_pandas(profile(model=model, input_size=input_size))
        if attr not in data_frame.columns:
            print('attr not found, select from:')
            print(data_frame.columns)
            return
        fig.add_trace(go.Bar(y=data_frame[attr], x=data_frame['Name'], name=name))
        

    fig.update_layout(xaxis_title="function call")
    fig.update_layout(yaxis_title=attr)
    fig.show()

def visualize_profile_attr(exps, attr='CPU time', sort_by="CPU total", plot_first=10, plot_single=False, show_operations=True, average=False, ensemble_size=4):

    """
    INPUT -- Models -- A dictionary, key: name, value: model
    OUTPUT -- A graph
    """

    fig = go.Figure()

    data_to_plot = OrderedDict() 
    data_to_plot_name = OrderedDict()
    for exp in exps.keys():
        
        if exp[-1] == "s" and not plot_single:
            continue
        
        data_frame = exps[exp]
        data_frame.sort_values(by=[sort_by])
        
        if attr not in data_frame.columns:
            print('attr not found, select from:')
            print(data_frame.columns)
            return
        

        data_to_plot[exp] = data_frame[attr].head(plot_first)
        if exp[-1] == "h" or exp[-1] == "v":
            data_to_plot[exp] = data_to_plot[exp] * 4


        data_to_plot_name[exp] = data_frame['Name'].head(plot_first)

    if average:
        y = [np.mean(df) for df in data_to_plot.values()] 
        fig.add_trace(go.Bar(y=y, x=list(data_to_plot.keys())))
    
    else:
        for i in range(plot_first):
            y = [df.at[i] for df in data_to_plot.values()]
            label = [df.at[i] for df in data_to_plot_name.values()]
            fig.add_trace(go.Bar(y=y, x=list(data_to_plot.keys()),name=label[0], marker_color=green_scale[i]))
        


    fig.update_layout(yaxis_title=attr)
        
    fig.update_layout(barmode='relative')
    return fig

def profile_and_save(model, input_size=(128,3,32,32), filename='profile.csv'):
    print(input_size)
    table_to_pandas(profile(model=model, input_size=input_size, reps=1)).to_csv(filename)

def read_profiles(directory="", depths=[40], growth_rates=[12], ensemble_size = 4, dataset = "cifar_10", show_ensemble_size=False):

    """
        Given a directory containing profiles in CSV format, this function reads them and return a dictionary
        key : Experiment; Value: corresponding pandas frame
    """

    profiles = {}
    for depth in depths:
        for growth_rate in growth_rates:
            f_h = "_".join(["d", str(depth), "gr", str(growth_rate), "e", str(ensemble_size), dataset, "h"])
            f_v = "_".join(["d", str(depth), "gr", str(growth_rate), "e", str(ensemble_size), dataset, "v"])
            f_s = "_".join(["d", str(depth), "gr", str(growth_rate), "e", str(ensemble_size), dataset, "s"])


            key = "|".join([str(depth),str(growth_rate)])

            if show_ensemble_size:
                key += "|" +ensemble_size
            
            profiles[key + "|h"] = pd.read_csv(directory + f_h + ".csv", skipfooter=3)
            profiles[key + "|v"] = pd.read_csv(directory + f_v + ".csv", skipfooter=3)
            profiles[key + "|s"] = pd.read_csv(directory + f_s + ".csv", skipfooter=3)

    return profiles


