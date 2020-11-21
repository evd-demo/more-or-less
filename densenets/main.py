import train
import EvD_setup as setup
import fire
from beautifultable import BeautifulTable

def run_vary_growth_rate(exp_dir="", run_ensemble=True, depth=100, growth_rate=12, ensemble_size=4, s_epochs=100, e_epochs=100, batch_size=64, 
    efficient=False, run_single=True, dataset = "cifar10", num_trial=1, resume=False):

    # Check if the parameters are comparable or not
    if run_ensemble == True:
        
        # calculate the growth_rate for the ensemble networks
        ensemble_growth_rate, ratio_param = setup.get_ensemble_growth_rate(depth=depth, growth_rate=growth_rate, ensemble_size=ensemble_size)
        
        print("growth_rate:", growth_rate)
        print("ratio_param:", ratio_param)
        
        if abs(ratio_param - 1.0) > 0.15 and ensemble_size == 4:
            raise Exception("Parameters are not comparable:",ratio_param)
    
    # Create the experimental directory
    exp_dir = setup.create_experimental_directory(exp_dir+"_vgr")

    # Print the local variables of this function
    setup.print_locals(locals())

    if run_single == True:
        
        # Run a single deep model with the given configuration
        train.configure_and_train_denseNet(
            data="./runs/", 
            save=exp_dir, 
            depth=depth, 
            growth_rate=growth_rate, 
            efficient=efficient, 
            valid_size=False, 
            n_epochs=s_epochs, 
            batch_size=batch_size, 
            seed=None,
            dataset=dataset,
            num_trial=num_trial,
            resume=resume)

    if run_ensemble == True:
    
        # Setup and train an ensemble of DenseNets with the calculated growth_rate
        train.configure_and_train_denseNet_ensemble(
            data="./runs/", 
            save=exp_dir, 
            depth=depth, 
            growth_rate=ensemble_growth_rate, 
            efficient=efficient, 
            valid_size=False, 
            n_epochs=e_epochs, 
            batch_size=batch_size, 
            seed=None, 
            num_networks=ensemble_size, 
            dataset=dataset,
            num_trial=num_trial, 
            resume=resume)

# Note: There are a total of 3 blocks with (depth - 4)/6 dense layers in each block. 
# This is because each unit DenseLayer is composed of two conv layers.

# Note: If K is a factor of 2, the number of parameters are kinda comparable


def run_vary_depth(exp_dir="", run_ensemble=True, depth=100, growth_rate=12, ensemble_size=4, s_epochs=100, e_epochs=100, batch_size=64, 
    efficient=False, run_single=True, dataset="cifar10", num_trial=1, resume=False):

    # calculate the depth for the ensemble networks
    ensemble_depth, ratio_param = setup.get_ensemble_depth(depth=depth, growth_rate=growth_rate, ensemble_size=ensemble_size)
    
    print("depth:", ensemble_depth)
    print("ratio_param:", ratio_param)
    
    # Check if the parameters are comparable or not
    if abs(ratio_param - 1.0) > 0.2 and ensemble_size == 4:
        raise Exception("Parameters are not comparable:",ratio_param)

    # Print the local variables of this function
    setup.print_locals(locals())

    # Create the experimental directory
    exp_dir = setup.create_experimental_directory(exp_dir+"_vd")

    if run_single == True:
        
        # Run a single deep model with the given configuration
        train.configure_and_train_denseNet(
            data="./runs/", 
            save=exp_dir, 
            depth=depth, 
            growth_rate=growth_rate, 
            efficient=efficient, 
            valid_size=False, 
            n_epochs=s_epochs, 
            batch_size=batch_size, 
            seed=None,
            dataset=dataset,
            num_trial=num_trial, 
            resume=resume)

    if run_ensemble == True:

        # Setup and train an ensemble of DenseNets with the calculated growth_rate
        train.configure_and_train_denseNet_ensemble(
            data="./runs/", 
            save=exp_dir, 
            depth=ensemble_depth, 
            growth_rate=growth_rate, 
            efficient=efficient, 
            valid_size=False, 
            n_epochs=e_epochs, 
            batch_size=batch_size, 
            seed=None, 
            num_networks=ensemble_size,
            dataset=dataset, 
            num_trial=num_trial,
            resume=resume)

def run(exp_dir="", run_ensemble=True, depth=100, growth_rate=12, ensemble_size=4, n_epochs=100, batch_size=64, efficient=False, vary_depth = False, 
    vary_growth_rate = False, dataset="cifar10", num_trial=1, resume=False, equalize_time=False):
    
    # Print the local variables of this function
    setup.print_locals(locals())

    single_epochs =  n_epochs
    vertical_epochs = n_epochs
    horizontal_epochs = n_epochs

    if equalize_time:
        print("Equalizing time")
        single_epochs, vertical_epochs, horizontal_epochs = setup.get_epoch_number(depth, growth_rate, ensemble_size, dataset, batch_size, n_epochs)
    
    print(single_epochs, vertical_epochs, horizontal_epochs)

    if vary_depth == True and vary_growth_rate == True:
        run_vary_depth(exp_dir, run_ensemble, depth, growth_rate, ensemble_size, single_epochs, vertical_epochs, batch_size, efficient, True, dataset ,num_trial, resume)
        run_vary_growth_rate(exp_dir, run_ensemble, depth, growth_rate, ensemble_size, single_epochs, horizontal_epochs, batch_size, efficient, False, dataset ,num_trial, resume)
        return
    
    if vary_depth == True:
        run_vary_depth(exp_dir, run_ensemble, depth, growth_rate, ensemble_size, single_epochs, vertical_epochs, batch_size, efficient, False, dataset ,num_trial, resume)
    
    if vary_growth_rate == True:
        run_vary_growth_rate(exp_dir, run_ensemble, depth, growth_rate, ensemble_size, single_epochs, horizontal_epochs, batch_size, efficient, False, dataset ,num_trial, resume)
    
if __name__ == '__main__':
   fire.Fire(run)

