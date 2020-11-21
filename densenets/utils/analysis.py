import os
import re
import glob
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
from collections import OrderedDict
import itertools
import EvD_setup as setup

import torch
from models import DenseNet
import torch.nn as nn

###########################
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

###########################


def get_files(directory, gr_list=[8,10,12,14,16,18,20,24,28,32], d_list=[100], ensemble_size = 4, show_ensemble_size=True, dataset="cifar10", nameformat="gd", vary="", label=False, verbose=False):
    """
    Loads the file into a dictionary of pandas dataframe

    INPUTS
    directory -- the directory to look at while loading the files
    d_list -- the list containing the depths of networks to look at
    gr_list -- the list containing the growth rates of the networks to look at
    nameformat -- optionally gdedv (gr_*_d_*_e_*_dataset_v(gr or d))

    RETURN
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes
    """

    sub_dir = sorted(os.listdir(directory))
    data = OrderedDict()
    directories_read = 0
    files_read_per_directory = 0
    
    ########################################################
    # Construct the suffix to append to the sub_directories
    ########################################################

    suffix = ""
    for f in nameformat[2:]:
        
        if f == 'e':
            suffix += "_e_" + str(ensemble_size)
        
        if f == 'd':
            suffix += "_" + dataset

        if f == 'v':
            suffix += "_v"+ vary
    
    ########################################################
    # Read files from all the directories and store them 
    # in a dictionary
    ########################################################
    
    for gr, depth in list(itertools.product(gr_list, d_list)):
        

        directory_name = "gr_"+str(gr)+"_d_"+str(depth) + suffix
        
        # Formulate the key
        
        key = ""
        if (len(d_list) == 1 or vary == "d") and label == True:
            key = "v | " + key
        
        elif (len(gr_list) == 1 or vary == "gr") and label == True:
            key = "h | " + key
        
        else:
            key = str(gr) + " | " +str(depth) 
        
        if show_ensemble_size:
            key = key +" | "+str(ensemble_size)
        
        if verbose:
            print(directory_name)
            print(key)
        
    
        data[key] = {}
        
        try:
            os.chdir(os.path.join(directory, directory_name))
            directories_read+=1
        except:
                print("not found!: " + directory_name)
        
        files = glob.glob("*.csv")
        
        for f in files:
            data[key][f.replace(".csv", "")] = pd.read_csv(f,index_col=False)
            files_read_per_directory += 1
        os.chdir(directory)
        
        #  (UGLY CODE!!!) because we don't have a single.csv for vary_gr for 
        # some of the experiments
        if vary == "gr":
            directory_name=directory_name[:-2] + "d"
            os.chdir(os.path.join(directory, directory_name))
            data[key]["single"] = pd.read_csv("single.csv",index_col=False)
            os.chdir(directory)
            files_read_per_directory += 1 

    # Not that ugly, but kinda ugly
    temp_data = {}
    
    ########################################################
    # Read the file to extract num_param to add to the key
    ########################################################

    for exp in data.keys():
        if "test_error" in data[exp]['single'].keys():
            num_param = str(round(data[exp]['single']["test_error"][0],2))
        else:
            num_param = str(round(list(data[exp]['single']['num_param'])[0],2))
        temp_data[exp+" | "+num_param] = data[exp]
    
    data = temp_data
    print("Read {}\n found {} sub directories\n containing {} files each".format(directory, directories_read, files_read_per_directory/directories_read ))
    return data

def get_metric_comparison(data_list, metric="valid_error", format="shvb"):

    """
    Gets a specified metric from the all files in the data dictionary
    
    INPUTS
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes
    metric -- the metric to extract
    format -- the format in which to extract it
    
    RETURN
    result -- a dictionary, keys: growth rate or depth; value: the metric in the specified format
    """

    ########################################################
    # Get the keys to iterate over, the keys represent the 
    # experiments. Intersect, to find common experiments
    ########################################################

    # print(data_list.keys())
    
    keys = []
    i=0
    for exp_type in data_list.keys():
        keys.append(data_list[exp_type].keys())
        print("{} has {} experiments".format(exp_type,len(keys[i])))
        i+=1
    
    keys = intersection(keys[0],keys[1])
    print("intersection has {}".format(len(keys)))

    result = pd.DataFrame(columns=keys)
    
    ########################################################
    # Go over the experiments and extract metric for h, s, 
    # and v
    ########################################################

    for exp in keys:
        metrics = {}
        
        for exp_type in data_list.keys():
            if exp_type[-1] == "h":
                metrics["h"] = np.array(1-data_list[exp_type][exp]['ensemble_results'][metric])
            
            if exp_type[-1] == "v":
                metrics["v"] = np.array(1-data_list[exp_type][exp]['ensemble_results'][metric])
            
            if "s" in metrics.keys():
                metrics["s"] += np.array(1-data_list[exp_type][exp]['single'][metric])
            else:
                metrics["s"] = np.array(1-data_list[exp_type][exp]['single'][metric])
            
        s = metrics["s"]/2.0
        h = metrics["h"]
        v = metrics["v"]

        max_len = max(len(s), len(h), len(v))
        if len(s) + len(h) + len(v) != 3*max_len:
            print("All experiments not equal, s:{}, h:{}, v:{}".format(len(s), len(h), len(v)))
            s = np.append(s,np.ones(max_len - len(s))*s[-1])
            h = np.append(h,np.ones(max_len - len(h))*h[-1])
            v = np.append(v,np.ones(max_len - len(v))*v[-1])
        
        assert len(s) + len(h) + len(v) == 3*max_len
        
        if format == "shvb":
            result[exp] =  np.logical_and(s>h, s>v) * 0 # single best
            result[exp] += np.logical_and(h>s, v<s) * 0.5 # Only h
            result[exp] += np.logical_and(v>s, h<s) * 1 # only v
            result[exp] +=  np.logical_and(v>s, h>s) * 1.5 # both best
        if format == "shv":
            result[exp] =  np.logical_and(s>h, s>v) * 0 # single best
            result[exp] += np.logical_and(h>s, h>v) * 0.5 # h best
            result[exp] += np.logical_and(v>s, v>h) * 1 # v best
        # result[exp] +=  np.logical_and(v>s, h>s) * 1.5 # both best

        
    
    return result

def get_metric_accuracy(data_list, metric="valid_error", format="shvb", maximum=False):

    """
    Gets a specified metric from the all files in the data dictionary
    
    INPUTS
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes
    metric -- the metric to extract
    format -- the format in which to extract it
    
    RETURN
    result -- a dictionary, keys: growth rate or depth; value: the metric in the specified format
    """

    ########################################################
    # Get the keys to iterate over, the keys represent the 
    # experiments. Intersect, to find common experiments
    ########################################################

    # print(data_list.keys())
    
    keys = []
    i=0
    for exp_type in data_list.keys():
        keys.append(data_list[exp_type].keys())
        print("{} has {} experiments".format(exp_type,len(keys[i])))
        i+=1
    
    keys = intersection(keys[0],keys[1])
    print("intersection has {}".format(len(keys)))

    result = OrderedDict()
    
    ########################################################
    # Go over the experiments and extract metric for h, s, 
    # and v
    ########################################################
    result['s'] = OrderedDict()
    result['h'] = OrderedDict()
    result['v'] = OrderedDict()
    result['hn'] = OrderedDict()
    result['vn'] = OrderedDict()

    
    for exp in keys:
        metrics = {}
        
        
        for exp_type in data_list.keys():
            if exp_type[-1] == "h":
                metrics["h"] = np.array(1-data_list[exp_type][exp]['ensemble_results'][metric])
                
                network_acc = []
                for network in data_list[exp_type][exp].keys():
                    
                    if network =='ensemble_results':
                        continue
                    network_acc.append(data_list[exp_type][exp][network][metric])
                
                metrics["hn"] = 1-np.mean(network_acc)
            
            if exp_type[-1] == "v":
                metrics["v"] = np.array(1-data_list[exp_type][exp]['ensemble_results'][metric])

                network_acc = []
                for network in data_list[exp_type][exp].keys():
                    
                    if network =='ensemble_results':
                        continue
                    network_acc.append(data_list[exp_type][exp][network][metric])
                
                metrics["vn"] = 1-np.mean(network_acc)
            
            if "s" in metrics.keys():
                metrics["s"] += np.array(1-data_list[exp_type][exp]['single'][metric])
            else:
                metrics["s"] = np.array(1-data_list[exp_type][exp]['single'][metric])
            
        s = metrics["s"]/2.0
        h = metrics["h"]
        v = metrics["v"]

        if maximum:
            result['s'][exp] = max(s)
            result['h'][exp] = max(h)
            result['v'][exp] = max(v)
        else:
            result['s'][exp] = s[-1]
            result['h'][exp] = h[-1]
            result['v'][exp] = v[-1]
        
        result['hn'][exp] = metrics['hn']
        result['vn'][exp] = metrics['vn']
    
    return result

def get_metric_time_to_accuracy(data_list, metric="valid_error", format="shvb"):

    """
    Gets a specified metric from the all files in the data dictionary
    
    INPUTS
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes
    metric -- the metric to extract
    format -- the format in which to extract it
    
    RETURN
    result -- a dictionary, keys: growth rate or depth; value: the metric in the specified format
    """

    ########################################################
    # Get the keys to iterate over, the keys represent the 
    # experiments. Intersect, to find common experiments
    ########################################################

    # print(data_list.keys())
    
    keys = []
    i=0
    for exp_type in data_list.keys():
        keys.append(data_list[exp_type].keys())
        print("{} has {} experiments".format(exp_type,len(keys[i])))
        i+=1
    
    keys = intersection(keys[0],keys[1])
    print("intersection has {}".format(len(keys)))

    result = OrderedDict()
    
    ########################################################
    # Go over the experiments and extract metric for h, s, 
    # and v
    ########################################################
    result['s'] = OrderedDict()
    result['h'] = OrderedDict()
    result['v'] = OrderedDict()
    
    for exp in keys:
        metrics = {}
        time = {}
        
        
        for exp_type in data_list.keys():
            if exp_type[-1] == "h":
                metrics["h"] = np.array(1-data_list[exp_type][exp]['ensemble_results'][metric])
                time["h"] = np.mean(data_list[exp_type][exp]['ensemble_results']["train_time"])
            
            if exp_type[-1] == "v":
                metrics["v"] = np.array(1-data_list[exp_type][exp]['ensemble_results'][metric])
                time["v"] = np.mean(data_list[exp_type][exp]['ensemble_results']["train_time"])
            
            if "s" in metrics.keys():
                metrics["s"] += np.array(1-data_list[exp_type][exp]['single'][metric])
            else:
                metrics["s"] = np.array(1-data_list[exp_type][exp]['single'][metric])
                time["s"] = np.mean(data_list[exp_type][exp]['single']["train_time"])
            
        s = metrics["s"]/2.0
        h = metrics["h"]
        v = metrics["v"]

        
        epochs_h = np.where(h > max(s))[0]
        epochs_v = np.where(v > max(s))[0]

        

        # print(exp, epochs_h, epochs_v)

        result['s'][exp] = len(s)* time['s'] / 60
        if len(epochs_h) == 0:
            result['h'][exp] = 0
        else:
            result['h'][exp] = epochs_h[0] * time['h'] /60
        
        if len(epochs_v) == 0:
            result['v'][exp] = 0
        else:
            result['v'][exp] = epochs_v[0] * time['v'] /60

    
    return result



def get_metric(data, metric="valid_error", format="s:e"):

    """
    Gets a specified metric from the all files in the data dictionary
    
    INPUTS
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes
    metric -- the metric to extract
    format -- the format in which to extract it
    
    RETURN
    result -- a dictionary, keys: growth rate or depth; value: the metric in the specified format
    """

    result = pd.DataFrame(columns = data.keys())

    for exp in data.keys():
        
        print(exp)
        
        num_epoch = min([len(data[exp]['single.csv'][metric]),len(1-data[exp]['ensemble_results.csv'][metric])])
        # print(num_epoch)
        
        if format == "e>s":
            result[exp] = (np.array((1-data[exp]['single.csv'][metric][:num_epoch]))<=np.array((1-data[exp]['ensemble_results.csv'][metric][:num_epoch])))*1
        
        if format == "s:e":
            result[exp] = (1-data[exp]['single.csv'][metric])/(1-data[exp]['ensemble_results.csv'][metric])
        
        if format == "e:s":
            result[exp] = 1/((1-data[exp]['single.csv'][metric])/(1-data[exp]['ensemble_results.csv'][metric]))
        
        if format == "e":
            result[exp] = 1-data[exp]['ensemble_results.csv'][metric]
        
        if format == "s":
            result[exp] = 1-data[exp]['single.csv'][metric]
    
    return result


        
        
        
            



    #         result[exp] = (np.array((1-data_list[exp]['single.csv'][metric]))<=np.array((1-data[exp]['ensemble_results.csv'][metric])))*1


    #     data = data_list[exp_type]

    #     for exp in data.keys():
        
    #     if format == "e>s":
    #         result[exp] = (np.array((1-data[exp]['single.csv'][metric]))<=np.array((1-data[exp]['ensemble_results.csv'][metric])))*1
        
    #     if format == "s:e":
    #         result[exp] = (1-data[exp]['single.csv'][metric])/(1-data[exp]['ensemble_results.csv'][metric])
        
    #     if format == "e:s":
    #         result[exp] = 1/((1-data[exp]['single.csv'][metric])/(1-data[exp]['ensemble_results.csv'][metric]))
        
    #     if format == "e":
    #         result[exp] = 1-data[exp]['ensemble_results.csv'][metric]
        
    #     if format == "s":
    #         result[exp] = 1-data[exp]['single.csv'][metric]
    
    # return result

def get_acc(data, average=False):

    """
    INPUT
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes

    RETURN
    A list of accuracies corresponding to s,e, and the ensemble networks
    """

    result = {}

    for exp in data.keys():
        exp_data = data[exp]
        for model in exp_data.keys():

            key = model

            # if any(i.isdigit() for i in model):
            #     key = 'ensemble'
            
            if key not in result.keys():
                result[key] = []
            result[key].append(1-exp_data[model]['valid_error'][119])

        # acc_s.append(1-np.min(data[exp]['single.csv']['valid_error']))
        # acc_e.append(1-np.min(data[exp]['ensemble_results.csv']['valid_error']))
        # acc_n_0.append(1-data[exp]['0_results.csv']['valid_error'][119])
        # acc_n_1.append(1-data[exp]['1_results.csv']['valid_error'][119])
        # acc_n_2.append(1-data[exp]['2_results.csv']['valid_error'][119])
        # acc_n_3.append(1-data[exp]['3_results.csv']['valid_error'][119])
    return result

def get_per_epoch_time(data):
    '''
    INPUT
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes

    RETURN
    A dictionary, keys: single and ensemble, value: corresponding training time
    *_n network time
    *d data loading time
    '''
    time = {}
    time['single'] = []
    time['ensemble'] = []
    time['single_d'] = []
    time['ensemble_d'] = []
    time['single_n'] = []
    time['ensemble_n'] = []

    for exp in data.keys():
        print(exp)
        time['single'].append(np.mean(data[exp]['single.csv']['train_time']))
        time['ensemble'].append(np.mean(data[exp]['ensemble_results.csv']['train_time']))
        if 'data_time' in data[exp]['single.csv'].keys() and 'data_time' in data[exp]['ensemble_results.csv'].keys():
            time['single_d'].append(np.mean(data[exp]['single.csv']['data_time']))
            time['ensemble_d'].append(np.mean(data[exp]['ensemble_results.csv']['data_time']))
            time['single_n'].append(np.mean(data[exp]['single.csv']['train_time']) - np.mean(data[exp]['single.csv']['data_time']))
            time['ensemble_n'].append(np.mean(data[exp]['ensemble_results.csv']['train_time']) - np.mean(data[exp]['ensemble_results.csv']['data_time']))
        else:
            time['single_d'].append(0)
            time['ensemble_d'].append(0)
            time['single_n'].append(0)
            time['ensemble_n'].append(0)

    return time

def get_time_to_accuracy(data):
    '''
    INPUT
    data -- a dictionary, keys: growth rate or depth; value: pandas dataframes

    RETURN
    A dictionary, keys: single and ensemble, value: corresponding training time
    '''
    
    ensemble_time = []
    single_time = []
    
    for exp in data.keys():
        best_acc_single = np.min(data[exp]['single']['valid_error'])
        epochs = np.where(data[exp]['ensemble_results']['valid_error'] < best_acc_single)[0]
        if len(epochs) == 0:
            ensemble_time.append(0 * data[exp]['ensemble_results']['train_time'][0])
        else:
            ensemble_time.append(epochs[0] * data[exp]['ensemble_results']['train_time'][0])
        
        single_time.append(119 * data[exp]['single']['train_time'][0])

    return single_time, ensemble_time

def get_num_param(data, format="str"):
    """
        Get the number of parameters associated with every data file

        INPUT
        data -- a dictionary, keys: growth rate or depth; value: pandas dataframes

        OUTPUT
        List, where every position is the number of parameters
    """

    num_param = []
    for exp in data.keys():
        if format=="str":
            if "test_error" in data[exp]['single'].keys():
                num_param.append(str(round(data[exp]['single']["test_error"][0],2))) # TODO Fix this, for now this wrongfully refers to num param
            else:
                num_param.append(str(round(data[exp]['single']["num_param"],2))) 
            
        
        sorted_order = np.argsort(num_param,kind='stable')

    return sorted_order, num_param

def sort_data(data, posn=0):

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

def get_ensemble_param(gr_list, d_list, ensemble_size, verbose=False):
    
    """
        Given a list of growth rates (gr_list) and depths (d), 
        return the list of single depth/gr, vertical depth/gr
        and horizonal depth/gr
    """
    
    s_d = []
    s_gr = []
    v_d = []
    v_gr = []
    h_d = []
    h_gr = []
    
    for gr, depth in list(itertools.product(gr_list, d_list)):
        vertical_depth, _ = setup.get_ensemble_depth(depth, gr, ensemble_size)
        horizontal_gr, _ = setup.get_ensemble_growth_rate(depth, gr, ensemble_size)
        if verbose:
            print("s: d: {0}, gr {1}".format(depth,gr))
            print("v: d: {0}, gr {1}".format(vertical_depth,gr))
            print("h: d: {0}, gr {1}".format(depth,horizontal_gr))
        
        s_d.append(depth)
        s_gr.append(gr)
        v_d.append(vertical_depth)
        v_gr.append(gr)
        h_d.append(depth)
        h_gr.append(horizontal_gr)
        
    return s_d, s_gr, v_d, v_gr, h_d, h_gr