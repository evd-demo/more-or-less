import os
import numpy as np
import pandas as pd
np.set_printoptions(precision=2)
pd.options.display.precision = 2
pd.set_option('display.precision', 2)
# 
import plotly
import plotly.graph_objs as go
import plotly.express as px
# 
from collections import defaultdict
from utils.results import Results, Experiment
from utils.dashboarding import load_experiment, plot_dimension
from collections import OrderedDict
from models import wrn

import sys
sys.path.append('../')
from models import *

import glob
import itertools


def get_accuracy_training_time(directory, param, ens_size, valid=True):

    """
    Get the accuracy and training time from the pickle files in the directory. 

    directory -- the directory to read the pickle files from
    param -- range of parameters to read 
    ens_size -- ensemble size
    valid -- whether to include the validation accuracy or the training accuracy

    """
    os.chdir(directory)
    files = glob.glob("*.pkl")

    result = {}
    for conf in list(itertools.product(param, ens_size)):
        f = str(conf[0])+"M_"+str(conf[1])+".pkl"
        print(f)
        exp = load_experiment(f)
        result[f] = {}
        # 
        result[f]['single'] = [ conf[0], conf[1], max(exp.data['single'].valid_accy), sum(exp.data['single'].time)/60/60]
        result[f]['ensemble'] = [ conf[0], conf[1], max(exp.data['ensemble'].data['ensemble'].valid_accy), sum(exp.data['ensemble'].data['ensemble'].time)/60/60 ]

        average_accuracy_ensembles = 0
        accuracies = []
        for k in ['net_1','net_2','net_3','net_4']:
            accuracies.append(exp.data['ensemble'].data[k].valid_accy[-1])
        min_accuracy_ensembles = np.min(accuracies)
        result[f]['per_ensemble_network'] = [ conf[0], conf[1], min_accuracy_ensembles, sum(exp.data['ensemble'].data['net_1'].time)/60/60]

    return result

def get_accuracy_training_time_resnet(directory, depth, width_param, ens_size, valid=True, n_epochs=250, metric="s:e", suffix="fixed_depth"):

    """
    Get the accuracy and training time from the pickle files in the directory. 

    directory -- the directory to read the pickle files from
    param -- range of parameters to read 
    ens_size -- ensemble size
    valid -- whether to include the validation accuracy or the training accuracy

    """
    os.chdir(directory)
    files = glob.glob("*.pkl")

    result = {}
    for conf in list(itertools.product(depth, width_param, ens_size)):
        f = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_" + suffix + ".pkl"
        print(f)

        exp = load_experiment(f)
        result[f] = {}
        # 
        result[f]['single'] = [ conf[0], conf[1], max(exp.data['single'].valid_accy), sum(exp.data['single'].time)/60/60]
        result[f]['ensemble'] = [ conf[0], conf[1], max(exp.data['ensemble'].data['ensemble'].valid_accy), sum(exp.data['ensemble'].data['ensemble'].time)/60/60 ]

        average_accuracy_ensembles = 0
        accuracies = []
        for k in ['net_1','net_2','net_3','net_4']:
            accuracies.append(exp.data['ensemble'].data[k].valid_accy[-1])
        min_accuracy_ensembles = np.min(accuracies)
        result[f]['per_ensemble_network'] = [ conf[0], conf[1], min_accuracy_ensembles, sum(exp.data['ensemble'].data['net_1'].time)/60/60]

    return result

def get_data(file_name, networks=["net_1", "net_2", "net_3", "net_4"], single=True, ensemble=True, attr='valid_accy'):

    """
    Returns the data stored in a utils.result object

    file_name -- the pickle file to read
    networks -- the ensemble networks to include
    single -- whether to include single network accuracy 
    ensemble -- whether to include the ensemble accuracy
    attr -- the attribute to get
    """
    result = {}
    
    exp = load_experiment(file_name)
    
    if single:
        result['single'] = getattr(exp.data['single'], attr)
    
    if ensemble:
        result['ensemble'] = getattr(exp.data['ensemble'].data['ensemble'], attr)
    
    for network in networks:
        result[network] = getattr(exp.data['ensemble'].data[network], attr)
    
    return result


def get_epoch_accuracy(directory, param, ens_size, valid=True, n_epochs=400, metric="s:e"):
    
    """
    Returns result of size [experiments, number of epochs] containing the metric specified

    directory -- the directory to read the files from
    param -- range of param number
    ens_size -- the size of ensembles
    valid -- whether to report the validation accuracy or the training accuracy
    n_epochs -- how many epochs to include
    metric -- the metric that the result tensor will contain
    """
    os.chdir(directory)
    files = glob.glob("*.pkl")

    result = np.zeros([len(param), n_epochs])
    i=0
    for conf in list(itertools.product(param, ens_size)):
        f = str(conf[0])+"M_"+str(conf[1])+".pkl"
        exp = load_experiment(f)
        
        if valid == True:
            single = np.array(exp.data['single'].valid_accy)
            ensemble = np.array(exp.data['ensemble'].data['ensemble'].valid_accy)
        else:
            single = np.array(exp.data['single'].train_accy)
            ensemble = np.array(exp.data['ensemble'].data['ensemble'].train_accy)

        
        if metric == "s:e":
            result[i,:] = (single / ensemble)
        
        if metric == "e:s":
            result[i,:] = (ensemble / single)
        
        if metric == "s>e":
            result[i,:] = (single > ensemble)
        
        if metric == "e>s":
            result[i,:] = (single < ensemble)
        
        
        if metric == "s":
            result[i,:] = single
        
        if metric == "e":
            result[i,:] = ensemble

        i+=1
    return result.T

def get_epoch_accuracy_resnet(directory, depth, width_param, ens_size, valid=True, n_epochs=250, metric="s:e", suffix=["fixed_depth"], architecture="resnet"):
    
    """
    Returns result of size [experiments, number of epochs] containing the metric specified

    directory -- the directory to read the files from
    param -- range of param number
    ens_size -- the size of ensembles
    valid -- whether to report the validation accuracy or the training accuracy
    n_epochs -- how many epochs to include
    metric -- the metric that the result tensor will contain
    """
    os.chdir(directory)
    files = glob.glob("*.pkl")

    result = np.zeros([len(width_param)*len(depth)*len(ens_size)*len(suffix), n_epochs])
    keys = []
    i=0
    for conf in list(itertools.product(depth, width_param, ens_size, suffix)):
        f = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_" + str(conf[3]) + ".pkl"
        
        if architecture == "resnet":
            params = resnets.count_parameters(resnets.make_resnet(depth=conf[0], width_parameter = conf[1]))/100000.0
        elif architecture == "wrn":
            params = resnets.count_parameters(wrn.WideResNet(depth=conf[0], widen_factor=conf[1], num_classes=10))


        
        keys.append("|".join([str(conf[1]),str(conf[0]),str(conf[2]),conf[3][0],str(params)]))
        
        print(f)
        exp = load_experiment(f)
        
        if valid == True:
            single = np.array(exp.data['single'].valid_accy)
            ensemble = np.array(exp.data['ensemble'].data['ensemble'].valid_accy)
        else:
            single = np.array(exp.data['single'].train_accy)
            ensemble = np.array(exp.data['ensemble'].data['ensemble'].train_accy)

        
        if metric == "s:e":
            result[i,:] = (single / ensemble)
        
        if metric == "e:s":
            result[i,:] = (ensemble / single)
        
        if metric == "s>e":
            result[i,:] = (single > ensemble)
        
        if metric == "e>s":
            result[i,:] = (single < ensemble)
        
        
        if metric == "s":
            result[i,:] = single
        
        if metric == "e":
            result[i,:] = ensemble

        i+=1
    return keys, result.T

def get_epoch_accuracy_resnet_combined(directory, depth, width_param, ens_size, valid=True, n_epochs=250, metric="s:e", suffix=["fixed_depth"], show_ensemble=False, architecture="resnet"):
    
    """
    Returns result of size [experiments, number of epochs] containing the metric specified

    directory -- the directory to read the files from
    param -- range of param number
    ens_size -- the size of ensembles
    valid -- whether to report the validation accuracy or the training accuracy
    n_epochs -- how many epochs to include
    metric -- the metric that the result tensor will contain
    """
    os.chdir(directory)
    files = glob.glob("*.pkl")

    result = np.zeros([len(width_param)*len(depth)*len(ens_size), n_epochs])
    keys = []
    i=0
    for conf in list(itertools.product(depth, width_param, ens_size)):
        f_v = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_vertical"  + ".pkl"
        f_h = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_horizontal"  + ".pkl"
        
        if architecture == "resnet":
            params = resnets.count_parameters(resnets.make_resnet(depth=conf[0], width_parameter = conf[1]))
        elif architecture == "wrn":
            params = resnets.count_parameters(wrn.WideResNet(depth=conf[0], widen_factor=conf[1], num_classes=10))
        
        params = round(params,2)
        
        if show_ensemble:
            keys.append(" | ".join([str(conf[1]),str(conf[0]),str(conf[2]),str(params)]))
        else:
            keys.append(" | ".join([str(conf[1]),str(conf[0]),str(params)]))
        
        exp_v = load_experiment(f_v)
        exp_h = load_experiment(f_h)
        
        
        s = np.array(exp_v.data['single'].valid_accy)
        h = np.array(exp_h.data['ensemble'].data['ensemble'].valid_accy)
        v = np.array(exp_v.data['ensemble'].data['ensemble'].valid_accy)

        result[i,:] =  np.logical_and(s>h, s>v) * 0 # single best
        result[i,:] += np.logical_and(h>s, v<s) * 0.5 # Only h
        result[i,:] += np.logical_and(v>s, h<s) * 1 # only v
        result[i,:] +=  np.logical_and(h>s, v>s) * 1.5 # both best


        i+=1
    return keys, result.T

def get_final_accuracy(directory, depth, width_param, ens_size, valid=True, n_epochs=250, metric="s:e", suffix=["fixed_depth"], show_ensemble=False, maximum=True, architecture="resnet"):
    
    """
    Returns result of size [experiments, number of epochs] containing the metric specified

    directory -- the directory to read the files from
    param -- range of param number
    ens_size -- the size of ensembles
    valid -- whether to report the validation accuracy or the training accuracy
    n_epochs -- how many epochs to include
    metric -- the metric that the result tensor will contain
    """
    os.chdir(directory)
    files = glob.glob("*.pkl")

    result = {}
    result['s'] = OrderedDict()
    result['h'] = OrderedDict()
    result['v'] = OrderedDict()
    result['hn'] = OrderedDict()
    result['vn'] = OrderedDict()

    i=0
    for conf in list(itertools.product(depth, width_param, ens_size)):
        f_v = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_vertical"  + ".pkl"
        f_h = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_horizontal"  + ".pkl"
        
        if architecture == "resnet":
            params = resnets.count_parameters(resnets.make_resnet(depth=conf[0], width_parameter = conf[1]))
        elif architecture == "wrn":
            params = resnets.count_parameters(wrn.WideResNet(depth=conf[0], widen_factor=conf[1], num_classes=10))
        
        params = round(params,2)

        if show_ensemble:
            key = " | ".join([str(conf[1]),str(conf[0]),str(conf[2]),str(params)])
        else:
            key = " | ".join([str(conf[1]),str(conf[0]),str(params)])
        
        exp_v = load_experiment(f_v)
        exp_h = load_experiment(f_h)
        
        
        s = np.array(exp_v.data['single'].valid_accy)
        h = np.array(exp_h.data['ensemble'].data['ensemble'].valid_accy)
        v = np.array(exp_v.data['ensemble'].data['ensemble'].valid_accy)

        if maximum:
            result['s'][key] = max(s)
            result['h'][key] = max(h)
            result['v'][key] = max(v)
        else:
            result['s'][key] = s[-1]
            result['h'][key] = h[-1]
            result['v'][key] = v[-1]


        i+=1
    return result

def get_time_to_accuracy(directory, depth, width_param, ens_size, valid=True, n_epochs=250, metric="s:e", suffix=["fixed_depth"], show_ensemble=False, maximum=True, architecture="resnet"):
    
    """
    Returns result of size [experiments, number of epochs] containing the metric specified

    directory -- the directory to read the files from
    param -- range of param number
    ens_size -- the size of ensembles
    valid -- whether to report the validation accuracy or the training accuracy
    n_epochs -- how many epochs to include
    metric -- the metric that the result tensor will contain
    """
    os.chdir(directory)
    files = glob.glob("*.pkl")

    result = {}
    result['s'] = OrderedDict()
    result['h'] = OrderedDict()
    result['v'] = OrderedDict()
    result['hn'] = OrderedDict()
    result['vn'] = OrderedDict()

    i=0
    for conf in list(itertools.product(depth, width_param, ens_size)):
        f_v = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_vertical"  + ".pkl"
        f_h = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_horizontal"  + ".pkl"
        
        if architecture == "resnet":
            params = resnets.count_parameters(resnets.make_resnet(depth=conf[0], width_parameter = conf[1]))
        elif architecture == "wrn":
            params = resnets.count_parameters(wrn.WideResNet(depth=conf[0], widen_factor=conf[1], num_classes=10))
        
        params = round(params,2)
        
        if show_ensemble:
            key = " | ".join([str(conf[1]),str(conf[0]),str(conf[2]),str(params)])
        else:
            key = " | ".join([str(conf[1]),str(conf[0]),str(params)])
        
        exp_v = load_experiment(f_v)
        exp_h = load_experiment(f_h)
        
        # get accuracy
        s = np.array(exp_v.data['single'].valid_accy)
        h = np.array(exp_h.data['ensemble'].data['ensemble'].valid_accy)
        v = np.array(exp_v.data['ensemble'].data['ensemble'].valid_accy)

        # get per epoch time
        s_per_epoch_time = np.mean(exp_v.data['single'].time)
        h_per_epoch_time = np.mean(exp_h.data['ensemble'].data['ensemble'].time)
        v_per_epoch_time = np.mean(exp_v.data['ensemble'].data['ensemble'].time)

        epochs_h = np.where(h > max(s))[0]
        epochs_v = np.where(v > max(s))[0]
        
        result['s'][key] = len(s) * s_per_epoch_time
        if len(epochs_h) == 0:
            result['h'][key] = 0
        else:
            result['h'][key] = epochs_h[0] * h_per_epoch_time
        
        if len(epochs_v) == 0:
            result['v'][key] = 0
        else:
            result['v'][key] = epochs_v[0] * v_per_epoch_time

        i+=1
    return result

def get_epoch_accuracy_average(directories, param, ens_size, valid=True, n_epochs=400, metric="s:e"):

    """
    An extension of the above function that averages across multiple directories. The directories should have 
    identical pickle files
    """
    
    results = np.zeros([len(directories), n_epochs, len(param)])
    for i in range(len(directories)):
        print(directories[i])
        results[i,:,:] = get_epoch_accuracy(directories[i], param, ens_size, valid, n_epochs, metric)
    
    return np.mean(results, axis=0)


def plot_heatmap(directories, param_range, valid=True, metric="e:s"):
    
    """
    Plot a heatmap: x-axis: Number of parameters; y-axis: Number of epochs; z-axis: the specified metric

    directories -- List of directories to read from. They should contain identically named files that will be averaged
    across
    param_range -- The range of parameters to read from the directories
    valid -- whether to plot the validation accuracy or the training accuracy
    metric -- the metric to plot [s:e, e:s, e, s, s>e]
    """
    result = get_epoch_accuracy_average(directories, param_range, [4], valid = valid, metric=metric)

    fig = go.Figure(data=go.Heatmap(z=result, x=param_range, colorscale = px.colors.sequential.YlGn))
    fig.update_layout(xaxis_title="Number of parameters (M)")
    fig.update_layout(yaxis_title="Number of epochs")
    fig.update_layout(title=metric + " | test: " + str(valid))
    
    fig.show()

def plot_training_time(directory, param, ens_size, format="per_epoch"):

    os.chdir(directory)
    
    if format == "per_epoch":
        single_per_epoch_time = []
        ensemble_per_epoch_time = []

        for conf in list(itertools.product(param, ens_size)):
            
            f = str(conf[0])+"M_"+str(conf[1])+".pkl"
            time_data = get_data(f,attr="time")

            single_per_epoch_time.append(np.mean(time_data['single']))
            ensemble_per_epoch_time.append(np.mean(time_data['ensemble']))

        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(y=single_per_epoch_time, x=param,
                    name='single'))
        fig.add_trace(go.Bar(y=ensemble_per_epoch_time, x=param,
                    name='ensemble'))
        
        fig.update_layout(xaxis_title="Number of parameters (M)")
        fig.update_layout(yaxis_title="Per epoch training time (s)")

        fig.show()
    
    # if format == "convergence":
    #     for conf in list(itertools.product(param, ens_size)):
            
    #         f = str(conf[0])+"M_"+str(conf[1])+".pkl"
    #         time_data = get_data(f,attr="train_accy")

    #         single_per_epoch_time.append(np.mean(time_data['single']))
    #         ensemble_per_epoch_time.append(np.mean(time_data['ensemble']))


def get_param_num(depths, width_params, network_type='resnet'):

    param = []
    if network_type == 'resnet':

        for depth in depths:
            for width_param in width_params:
                param.append(resnets.count_parameters(resnets.make_resnet(depth=depth, width_parameter = width_param))/100000.0)
        return param


def sort_data(keys, data, posn=0):

    """
        Sort the data by a certain aspect

        INPUT
        data -- a dictionary, keys: v/h | gr | depth | num_param; value: pandas dataframes, posn -- the posn in the key to sort by 
            e.g., posn=1 means sort by gr

        OUTPUT
        the input data dictionary sorted
    """

    # extract the sub_key to sort by
    subkeys = []
    for key in keys:
        try:
            subkeys.append(float(key.split('|')[posn]))
        except:
            subkeys.append(key.split('|')[posn])
    
    # Get the sorted order according to the subkeys

    sorted_order = np.argsort(subkeys,kind='stable')
    
    # sort 
    data_sorted = np.zeros(data.shape)
    keys_sorted = []
    for i in range(len(sorted_order)):
        data_sorted[:,i] = data[:,sorted_order[i]]
        keys_sorted.append(keys[sorted_order[i]])

    return keys_sorted, data_sorted


def convert_to_csv(in_directory, out_directory, depth, width_param, ens_size, valid, suffix=["vertical","horizontal"]):

    
    files = glob.glob("*.pkl")

    result = np.zeros([len(width_param)*len(depth)*len(ens_size), n_epochs])
    keys = []
    
    i=0
    
    for conf in list(itertools.product(depth, width_param, ens_size)):
        
        params = resnets.count_parameters(resnets.make_resnet(depth=conf[0], width_parameter = conf[1]))/100000.0

        os.chdir(in_directory)
        f_v = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_vertical"  + ".pkl"
        

        columns = ["epoch", "train_time", "valid_error", "num_param"]
        
        
        for i in range(ensemble_size):
            
            ensemble_data_frame = pd.DataFrame(columns=columns)
        
        
        
        
        
        f_h = "d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_e_" + str(conf[2]) + "_horizontal"  + ".pkl"
        
        
        
        keys.append("|".join([str(conf[0]),str(conf[1]),str(conf[2]),str(params)]))
        
        exp_v = load_experiment(f_v)
        exp_h = load_experiment(f_h)
        
        
        s = np.array(exp_v.data['single'].valid_accy)
        h = np.array(exp_h.data['ensemble'].data['ensemble'].valid_accy)
        v = np.array(exp_v.data['ensemble'].data['ensemble'].valid_accy)

        os.chdir(out_directory)

        out_dir_v
        out_dir_h
        
        os.mkdir()
        os.mkdir("d_" + str(conf[0]) + "_w_" + str(conf[1]) + "_h")
    
def sort_data_dict(data, posn=0):

    """
        Sort the data by a certain aspect

        INPUT
        data -- a dictionary, keys: v/h | gr | depth | num_param; value: pandas dataframes, posn -- the posn in the key to sort by 
            e.g., posn=1 means sort by gr

        OUTPUT
        the input data dictionary sorted
    """

    # extract the sub_key to sort by
    subkeys = []
    for key in data.keys():
        try:
            subkeys.append(float(key.split('|')[posn]))
        except:
            subkeys.append(key.split('|')[posn])
    
    # Get the sorted order according to the subkeys

    sorted_order = np.argsort(subkeys,kind='stable')
    
    # sort 
    
    dk = list(data.keys())
    data_sorted = {dk[k]: data[dk[k]] for k in sorted_order}

    return data_sorted
    


