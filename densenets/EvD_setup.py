from models import DenseNet
from beautifultable import BeautifulTable
import numpy as np

# Helper function: Count number of parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# Helper function: given a grwoth rate and depth, count the number of parameters 
# of a DenseNet
def count_densenNet_param(growth_rate, depth):
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=10,
        small_inputs=True,
    )
    return count_parameters(model)

# Get the growth rate of a given ensemble
def get_ensemble_growth_rate(depth, growth_rate, ensemble_size):
    
    single_model_param = count_densenNet_param(growth_rate,depth)
    param_per_ensemble = single_model_param/ensemble_size

    # print("deep model: ",single_model_param/1e6, "M")
    # print("Param per ensemble (ideal): ", param_per_ensemble/1e6, "M")

    for i in range(1,growth_rate):
        network_e_param = count_densenNet_param(i,depth)
        if network_e_param > param_per_ensemble:
            
            current = count_densenNet_param(i,depth)
            previous = count_densenNet_param(i-1,depth)
            
            current_ratio = abs(1 - single_model_param/(ensemble_size*current))
            previous_ratio = abs(1-single_model_param/(ensemble_size*previous))

            if current_ratio < previous_ratio:
                return i, single_model_param*1.0/(ensemble_size*current)*1.0
            else:
                return i-1, single_model_param*1.0/(ensemble_size*previous)*1.0

# Get the depth  of a given ensemble
def get_ensemble_depth(depth, growth_rate, ensemble_size, mode="best"):
    
    single_model_param = count_densenNet_param(growth_rate,depth)
    param_per_ensemble = single_model_param*1.0/ensemble_size*1.0

    # print("deep model: ",single_model_param/1e6, "M")
    # print("Param per ensemble (ideal): ", param_per_ensemble/1e6, "M")

    # Get all depths that are valid
    all_depths = np.array(range(7,depth+1)) # 7 is the smalles valid number for depth
    valid_depths = all_depths[np.where((all_depths - 4) % 3 == 0)]
    # print(valid_depths)

    for i in range(0,len(valid_depths)):
        network_e_param = count_densenNet_param(growth_rate,valid_depths[i])
        
        if network_e_param > param_per_ensemble:
        
            current = count_densenNet_param(growth_rate,valid_depths[i])
            previous = count_densenNet_param(growth_rate,valid_depths[i-1])

            # print(current,previous)
            # print(valid_depths[i],valid_depths[i-1])
            
            current_ratio = abs(1 - (single_model_param*1.0/(ensemble_size*current*1.0)))
            previous_ratio = abs(1 - (single_model_param*1.0/(ensemble_size*previous*1.0)))

            # print(valid_depths[i], single_model_param*1.0/(ensemble_size*current)*1.0)
            # print(valid_depths[i-1], single_model_param*1.0/(ensemble_size*previous)*1.0)

            if current_ratio < previous_ratio:
                return valid_depths[i], single_model_param*1.0/(ensemble_size*current)*1.0
            else:
                return valid_depths[i-1], single_model_param*1.0/(ensemble_size*previous)*1.0

####################################################################################################################
import errno
import os
from datetime import datetime

# Create the experimental directory with current timestamp
def create_experimental_directory(exp_dir):

    # Create the experimental directory 
    
    # with current timestamp
    if exp_dir == "":
        mydir = os.path.join(
            os.getcwd(),"runs",
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(mydir)
        last = os.path.join(os.getcwd(),"runs","last")
        # if os.path.exists(last):
        #     os.remove(last)
        # os.symlink(mydir, last)
        return mydir
    
    # or with the given name: exp_dir
    else:
        mydir = os.path.join(
                os.getcwd(),"runs",exp_dir)
        os.makedirs(mydir, exist_ok=True)
        last = os.path.join(os.getcwd(),"runs","last")
        # if os.path.exists(last):
        #     os.remove(last)
        # os.symlink(mydir, last)
        return mydir

def print_locals(locals):
    bt = BeautifulTable()
    for argument in locals:
        bt.append_row([argument,locals[argument]])
    print(bt)

#############################################################################################################################
import data
from EvD_setup import get_ensemble_depth, get_ensemble_growth_rate
from train import AverageMeter
import torch
from models import DenseNet
import time

def estimate_epoch_time(models, train_loader, num_batches = 10000):
    
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in models]
    batch_time = AverageMeter()

    end = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):
        
        # create vaiables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # fake forward and backward pass
        for optimizer, model in zip(optimizers, models):
            if torch.cuda.is_available():
                model = model.cuda()
            
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time()-end)
        end = time.time()

        if num_batches <= batch_idx:
            break
    
    torch.cuda.empty_cache()
    
    return batch_time.avg * len(train_loader)
    
def get_epoch_number(depth, growth_rate, ensemble_size, dataset, batch_size=256, num_epochs=120):

    # get data set
    train_set, _, small_inputs, num_classes, _ = data.get_dataset(dataset, "./data/")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    
        
    # single
    single = [DenseNet(
        growth_rate=growth_rate,
        block_config=[(depth - 4) // 6 for _ in range(3)],
        num_classes=num_classes,
        small_inputs=True,
        efficient=small_inputs,
        compression=1.0
    )]
   
    # vertical 
    ensemble_depth, _ = get_ensemble_depth(depth, growth_rate, ensemble_size)
    vertical = [DenseNet(
        growth_rate=growth_rate,
        block_config=[(ensemble_depth - 4) // 6 for _ in range(3)],
        num_classes=num_classes,
        small_inputs=small_inputs,
        efficient=True,
        compression=1.0
    )]*ensemble_size

    # horizontal
    ensemble_growth_rate, _ = get_ensemble_growth_rate(depth, growth_rate, ensemble_size)
    horizontal = [DenseNet(
        growth_rate=ensemble_growth_rate,
        block_config=[(depth - 4) // 6 for _ in range(3)],
        num_classes=num_classes,
        small_inputs=small_inputs,
        efficient=True,
        compression=1.0
    )]*ensemble_size

    single_epoch_time = estimate_epoch_time(single, train_loader) # ?just to warm the cache? 
    vertical_epoch_time = estimate_epoch_time(vertical, train_loader)
    single_epoch_time = estimate_epoch_time(single, train_loader)
    horizontal_epoch_time = estimate_epoch_time(horizontal, train_loader)

    print("single: ", single_epoch_time)
    print("vertical: ", vertical_epoch_time)
    print("horizontal: ", horizontal_epoch_time)


    max_epoch_time = max(single_epoch_time, vertical_epoch_time, horizontal_epoch_time)
    
    single_epochs = (max_epoch_time / single_epoch_time) * num_epochs
    vertical_epochs = (max_epoch_time / vertical_epoch_time) * num_epochs
    horizontal_epochs = (max_epoch_time / horizontal_epoch_time) * num_epochs

    return single_epochs, vertical_epochs, horizontal_epochs