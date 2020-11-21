import glob
import pylab
import numpy as np
import pandas as pd
import fire
import EvD_setup as setup

def plot_directory(dir,attrs=['train_error', 'valid_error', 'train_time']):
    
    ensemble_file = dir + "ensemble_results.csv"
    # setup.print_locals(locals())
    network_files = glob.glob(dir+"*_results.csv")
    print(network_files)
    network_files.remove(ensemble_file)
    config_file = dir + "config.txt"

    ## Print the configuration file
    with open(config_file, 'r') as f:
            buffer = f.read()
            print(buffer)
   
    ## read the csv files into pandas frames
    network_data = {}
    for files in network_files:
        network_data[files] = pd.read_csv(files)
    
    ensemble_data = pd.read_csv(ensemble_file)

    ## Plot the requested attributes and save the files
    for attr in attrs:
        
        ## plot the time attribute
        if attr == "train_time":
            
            total_times = {}
            total_times["ensemble"] = sum(ensemble_data[attr])
            i=0
            
            for exp in network_data.keys():
                total_times["n_"+str(i)] = sum(network_data[exp][attr])
                i+=1
            
            pylab.bar(x=range(len(total_times.keys())), height=total_times.values())
            pylab.xticks(range(len(total_times.keys())),total_times.keys())
            
            pylab.savefig(dir+attr+".pdf")
            pylab.close()
            continue

        ## plot the error attributes
        pylab.plot(ensemble_data['epoch'],ensemble_data[attr], label='ensemble_'+attr)
        
        i=0
        for exp in network_data.keys():
            data = network_data[exp]
            pylab.plot(data['epoch'],data[attr], label=str(i)+'_'+attr)
            i+=1
        pylab.legend()
        pylab.xlabel("Number of Epochs")

        pylab.savefig(dir+attr+".pdf")
        pylab.close()

        
    
    # print(ensemble_data['train_time'])
    # print(network_data[network_data.keys()[0]]['train_time'])

# plot_directory("./results/",['train_error','valid_error'])

def compare(single_dir = "./runs/s/", ens_dirs = ["./runs/e4_vary_depth/", "./runs/e8_vary_depth/", "./runs/e12_vary_depth/", "./runs/e16_vary_depth/"], attrs= ['train_error', 'valid_error', 'train_time']):
    
    single_file = single_dir + "results.csv"
    single_data = pd.read_csv(single_file)

    ens_data = {}
    for e in ens_dirs:
        ens_data[e] = pd.read_csv(e + "ensemble_results.csv")

    for attr in attrs: 

        ## plot the time attribute
        if attr == "train_time":
            
            total_times = {}
            epoch_times = {}
            
            total_times["single"] = sum(single_data[attr].fillna(0))/3600.0
            epoch_times["single"] = np.mean(single_data[attr].fillna(0))/60.0
            
            for data in ens_data.keys():
                total_times[data] = sum(ens_data[data][attr])/3600.0
                epoch_times[data] = np.mean(ens_data[data][attr])/60.0
            
            pylab.bar(x=range(len(total_times.keys())), height=total_times.values())
           
            pylab.xticks(range(len(total_times.keys())),total_times.keys())
            pylab.ylabel(attr+ "_total (hrs)")
            
            pylab.savefig(single_dir+attr+"_total.pdf")
            pylab.close()

            pylab.bar(x=range(len(epoch_times.keys())), height=epoch_times.values())
           
            pylab.xticks(range(len(epoch_times.keys())),epoch_times.keys())
            pylab.ylabel(attr+ "_epoch (min)")
            
            pylab.savefig(single_dir+attr+"_epoch.pdf")
            pylab.close()
            continue

        # plot the single model
        pylab.plot(single_data['epoch'], single_data[attr], label="single")
        
        # plot the ensembles
        for data in ens_data.keys():
            pylab.plot(ens_data[data]['epoch'], ens_data[data][attr], label=data)
        
        # set labels
        pylab.xlabel("Number of Epochs")
        pylab.ylabel(attr)

        # set legend
        pylab.legend()
        
        # save figure and close
        pylab.savefig(single_dir+attr+".pdf")
        pylab.close()

def 


if __name__ == '__main__':
    # fire.Fire(plot_directory)
    fire.Fire(compare)