import fire
import os
import time
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from beautifultable import BeautifulTable
import numpy as np
import copy

# 
from models import DenseNet
from utils.imagenet_prep import * 
from data import *

# Helper function: Count number of parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# [W] Helper function that prints and save arguments given to a function
    # usage: print_and_save_arguments(locals(),<save_dir>)
def print_and_save_arguments(arguments, save_dir):
    table = BeautifulTable()
    for a in arguments:
        table.append_row([a,arguments[a]])
    
    ## TODO: Remove this ugly bit from the code
    # Get densenet configuration
    depth = arguments['depth']
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    model = DenseNet(
        growth_rate=arguments['growth_rate'],
        block_config=block_config,
        num_classes=10,
        small_inputs=True,
        efficient=False,
    ) 
    table.append_row(["Param_per_model (M)",count_parameters(model)/1000000.0])
    table.append_row(["CUDA", torch.cuda.is_available()])
    print(table)
    with open(os.path.join(save_dir, "config.txt"), 'a') as f:
            f.write(str(table))
    return

# TODO: Add support to record the average time per epoch
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def get_subsample_dataset(trainset, subset):
    trainsubset = copy.deepcopy(trainset)
    trainsubset.data = [trainsubset.data[index] for index in subset]
    trainsubset.targets = [trainsubset.targets[index] for index in subset]
    return trainsubset

def train_epoch(model, loader, scheduler, optimizer, epoch, n_epochs, print_freq=10, ensemble_num=0, total_ensembles=1):
    
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    error_top_5 = AverageMeter()

    end = time.time()
    # Model on train mode
    model.train()

    for batch_idx, (input, target) in enumerate(loader):
        
        # Create vaiables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        
        data_time.update(time.time() - end)
        
        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        error_top_5.update(error_top_k(output, target), batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Ens: [%d/%d]' % (ensemble_num+1,total_ensembles),
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time: %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss: %.4f (%.4f)' % (losses.val, losses.avg),
                'Error: %.4f (%.4f)' % (error.val, error.avg),
                'Error (top 5): %.4f (%.4f)' % (error_top_5.val, error.avg)
            ])
            print(res)

    # Return summary statistics
    return data_time.avg * len(loader), batch_time.avg * len(loader), losses.avg, error.avg, error_top_5.avg

def train_epoch_ensemble(ensemble_models, loader, schedulers, optimizers, epoch, n_epochs, print_freq=10, load_once = True):

    """
    Train all models in the given ensemble
    
    INPUT
        ensemble_models -- list of ensemble models to train
        loader -- the data loader to use
        optimizers -- list of optimizers to use
        epoch -- the current epoch
        n_epochs -- the total number of epochs
        print_freq -- how frequently to print
        load_once -- True: implies outer loop data, inner loop model and vice versa
    
    OUTPUT
        A list of metrics for all 4 ensemble models
    """
    
    ensemble_size = len(ensemble_models)
    
    # list of objects to keep track of metrics
    data_times = [AverageMeter() for _ in range(ensemble_size)]
    batch_times = [AverageMeter() for _ in range(ensemble_size)]
    losses = [AverageMeter() for _ in range(ensemble_size)]
    errors = [AverageMeter() for _ in range(ensemble_size)]

    # start the timer 
    end = time.time()
    
    # Outer loop: data, Inner loop: models
    if load_once:

        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
             
            for i in range(ensemble_size):
                data_times[i].update(time.time() - end)
                
                # Model on train mode
                model = ensemble_models[i]
                model.train()
        
                # compute output
                output = model(input)
                loss = torch.nn.functional.cross_entropy(output, target)

                # measure accuracy and record loss
                batch_size = target.size(0)
                _, pred = output.data.cpu().topk(1, dim=1)
                errors[i].update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
                losses[i].update(loss.item(), batch_size)

                # compute gradient and do SGD step
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                schedulers[i].step()

                # measure elapsed time
                batch_times[i].update(time.time() - end)
                end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    # 'Ens: [%d/%d]' % (ensemble_num+1,total_ensembles),
                    'Epoch: [%d/%d]' % (epoch + 1, n_epochs)
                #     'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                #     'Time: %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                #     'Loss: %.4f (%.4f)' % (losses.val, losses.avg),
                #     'Error: %.4f (%.4f)' % (error.val, error.avg)
                ])

                print(res)
    else:
        # Outer loop: model, Inner loop: data
        for i in range(ensemble_size):
            
            for batch_idx, (input, target) in enumerate(loader):
                # Create vaiables
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()
                
                data_times[i].update(time.time() - end)

                # Model on train mode
                model = ensemble_models[i]
                model.train()
        
                # compute output
                output = model(input)
                loss = torch.nn.functional.cross_entropy(output, target)

                # measure accuracy and record loss
                batch_size = target.size(0)
                _, pred = output.data.cpu().topk(1, dim=1)
                errors[i].update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
                losses[i].update(loss.item(), batch_size)

                # compute gradient and do SGD step
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

                # measure elapsed time
                batch_times[i].update(time.time() - end)
                end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    # 'Ens: [%d/%d]' % (ensemble_num+1,total_ensembles),
                    'Epoch: [%d/%d]' % (epoch + 1, n_epochs)
                #     'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                #     'Time: %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                #     'Loss: %.4f (%.4f)' % (losses.val, losses.avg),
                #     'Error: %.4f (%.4f)' % (error.val, error.avg)
                ])

                print(res)

    # Return summary statistics for all models
    return [data_times[i].avg * len(loader) for i in range(ensemble_size)], [batch_times[i].avg * len(loader) for i in range(ensemble_size)], [losses[i].avg for i in range(ensemble_size)], [errors[i].avg for i in range(ensemble_size)]

def test_epoch(model, loader, print_freq=10, is_test=True, ensemble_num=0, total_ensembles=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    error_top_5 = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            error_top_5.update(error_top_k(output,target), batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Ens: [%d/%d]' % (ensemble_num+1,total_ensembles),
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                    'Error (top 5) %.4f (%.4f)' % (error_top_5.val, error_top_5.avg),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg * len(loader), losses.avg, error.avg, error_top_5.avg

# [W] Taking input and the ensemble models, produces an output based on ensemble averaging
def get_ensemble_average(models, input, num_classes=10):
    
    num_classes = models[0].num_classes
    output = torch.zeros([input.shape[0],num_classes]).cuda()
    for m in models:
        output += m(input)
    return output/len(models)*1.0

# [W] Extension of the test_epoch function for ensembles
    #  TODO: Add support for other ensemble combination methods such as voting and super-learner.
def test_epoch_ensemble(models, loader, print_freq=10, is_test=True, ensemble_num=0, total_ensembles=1, method = "EA"):
    
    # object to keep track of time and accuracy
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    error_top_5 = AverageMeter()


    # Models on eval mode
    for m in models:
        m.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            if method == "EA":
                output = get_ensemble_average(models,input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)
            error_top_5.update(error_top_k(output,target), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Ens: [%d/%d]' % (ensemble_num+1,total_ensembles),
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg * len(loader), losses.avg, error.avg, error_top_5.avg

def train(model, train_set, valid_set, test_set, save, n_epochs=300,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None, test_freq=1, trial="", dataset="cifar10"):
    if seed is not None:
        torch.manual_seed(seed)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    if dataset == "cifar10" or dataset == "cifar100" or dataset == "svhn":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.25 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)
    elif dataset == "imagenet" or dataset == "imagenet32" :
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.33 * n_epochs, 0.66 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save, str(trial) + 'single.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,train_time,data_time,valid_loss,valid_error,valid_time,valid_error_top_5,num_param\n')
            
    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        data_time, train_time, train_loss, train_error, _ = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            scheduler=scheduler,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        # test based on test_freq
        if epoch % test_freq == 0:
            valid_time, valid_loss, valid_error, valid_error_top_5 = test_epoch(
                model=model_wrapper,
                loader=valid_loader if valid_loader else test_loader,
                is_test=(not valid_loader)
            )

            # Determine if model is the best
            if valid_loader and valid_error < best_error:
                best_error = valid_error
                print('New best error: %.4f' % best_error)
                torch.save(model.state_dict(), os.path.join(save, str(trial)+'single_model.dat'))
            else:
                torch.save(model.state_dict(), os.path.join(save, str(trial)+'single_model.dat'))

            # Log results
            with open(os.path.join(save, str(trial)+'single.csv'), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,\n' % (
                    (epoch + 1),
                    train_loss,
                    train_error,
                    train_time,
                    data_time,
                    valid_loss,
                    valid_error,
                    valid_time,
                    valid_error_top_5,
                    count_parameters(model)/1000000.0,
                ))

    # # Final test of model on test set
    # model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model).cuda()
    # test_results = test_epoch(
    #     model=model,
    #     loader=test_loader,
    #     is_test=True
    # )
    # _, _, test_error = test_results
    # with open(os.path.join(save, 'results.csv'), 'a') as f:
    #     f.write(',,,,,%0.5f\n' % (test_error))
    # print('Final test error: %.4f' % test_error)

# [W] Extension of train method for ensemble training 
def train_ensemble(models, train_set, valid_set, test_set, save, n_epochs=300,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None, test_freq=1, 
          trial=""):
    
    # TODO: Fix top_5 error for ensembles
    
    # initialize seed
    if seed is not None:
        torch.manual_seed(seed)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
    
    # Containers for ensembles
    optimizers = []
    schedulers = []
    model_wrappers = []
    best_errors = []

    # intialization for every ensemble network
    for i in range(len(models)):
        # Model on cuda
        if torch.cuda.is_available():
            models[i] = models[i].cuda()

        # Wrap model for multi-GPUs, if necessary
        model_wrappers.append(models[i]) 
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_wrappers[i] = torch.nn.DataParallel(models[i]).cuda()
        
        # Optimizer
        optimizers.append(torch.optim.SGD(model_wrappers[i].parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd))
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[0.5 * n_epochs, 0.75 * n_epochs],gamma=0.1))

        # Start log
        with open(os.path.join(save, str(trial) + str(i) + '_results.csv'), 'w') as f:
            f.write('epoch,train_loss,train_error,train_time,data_time,valid_loss,valid_error,valid_time,valid_error_top_5,num_param\n')
        
        # best errors
        best_errors.append(1)
    
    # Start log for ensemble
    with open(os.path.join(save, str(trial) + 'ensemble_results.csv'), 'w') as f:
            f.write('epoch,train_loss,train_error,train_time,data_time,valid_loss,valid_error,valid_time,valid_error_top_5,num_param\n')

    # Train all models
    for epoch in range(n_epochs):
        
        
        # Train all models for this epoch
        data_times, train_times, train_losses, train_errors = train_epoch_ensemble(
            ensemble_models=model_wrappers,
            loader=train_loader,
            schedulers=schedulers,
            optimizers=optimizers,
            epoch=epoch,
            n_epochs=n_epochs
        )

        # Test all models
        
        for i in range(len(models)):
            # test based on test_freq
            if epoch % test_freq == 0:
                valid_time, valid_loss, valid_error, valid_error_top_5 = test_epoch(
                    model=model_wrappers[i],
                    loader=valid_loader if valid_loader else test_loader,
                    is_test=(not valid_loader),
                    ensemble_num=i,
                    total_ensembles=len(models)
                )

                # Determine if model is the best, if it is, then save it
                if valid_loader and valid_error < best_errors[i]:
                    best_errors[i] = valid_error
                    print('New best error: %.4f' % best_errors[i])
                    torch.save(models[i].state_dict(), os.path.join(save, str(i) + str(trial) + '_model'))
                else:
                    torch.save(models[i].state_dict(), os.path.join(save, str(i) + str(trial) + '_model.dat'))

                # Log results
                with open(os.path.join(save, str(i) + str(trial) + '_results.csv'), 'a') as f:
                    f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,\n' % (
                        (epoch + 1),
                        train_losses[i],
                        train_errors[i],
                        train_times[i],
                        data_times[i],
                        valid_loss,
                        valid_error,
                        valid_time,
                        valid_error_top_5,
                        count_parameters(models[0])/1000000.0,
                    ))
    
        # Training loss on the entire ensemble
        _, train_loss, train_error, _ = test_epoch_ensemble(
                    models=model_wrappers,
                    loader=train_loader,
                    is_test=(not valid_loader),
                    ensemble_num=i,
                    total_ensembles=len(models)
        )

        # Valid loss on the entire ensemble
        valid_time, valid_loss, valid_error, valid_error_top_5 = test_epoch_ensemble(
                models=model_wrappers,
                loader=valid_loader if valid_loader else test_loader,
                is_test=(not valid_loader),
                ensemble_num=i,
                total_ensembles=len(models)
            )
        
        train_time = sum(train_times)
        data_time = sum(data_times)

        # Log results
        with open(os.path.join(save, str(trial) +'ensemble_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,\n' % (
                    (epoch + 1),
                    train_loss,
                    train_error,
                    train_time,
                    data_time,
                    valid_loss,
                    valid_error,
                    valid_time,
                    valid_error_top_5,
                    count_parameters(models[0])/1000000.0*len(models),
            ))
        

    #     # Final test of model on test set
    #     model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    #     if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #         model = torch.nn.DataParallel(model).cuda()
    #     test_results = test_epoch(
    #         model=model,
    #         loader=test_loader,
    #         is_test=True
    #     )
    # _, _, test_error = test_results
    # with open(os.path.join(save, 'results.csv'), 'a') as f:
    #     f.write(',,,,,%0.5f\n' % (test_error))
    # print('Final test error: %.4f' % test_error)

def configure_and_train_denseNet(data, save, depth=100, growth_rate=12, efficient=False, valid_size=5000,
         n_epochs=300, batch_size=64, seed=None, test_freq=1, dataset="cifar10", num_trial=1, 
         resume=False):
    
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.
    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)
        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)
        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """
    # Print and save arguments given to this function 
    print_and_save_arguments(locals(),save)

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    train_set, test_set, small_inputs, num_classes, grayscale = get_dataset(dataset, data)

    if valid_size:
        print("No validation set, define it first")
        return
        # valid_set = data_function(data, train=True, transform=test_transforms)
        # indices = torch.randperm(len(train_set))
        # train_indices = indices[:len(indices) - valid_size]
        # valid_indices = indices[len(indices) - valid_size:]
        # train_set = torch.utils.data.Subset(train_set, train_indices)
        # valid_set = torch.utils.data.Subset(valid_set, valid_indices)
    else:
        valid_set = None
        print("Validating on the test set")

    # Models
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=num_classes,
        small_inputs=small_inputs,
        efficient=efficient,
        grayscale=grayscale,
    )
    
    # print(model)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)


    if num_trial == 1:
        
        train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save,
          n_epochs=n_epochs, batch_size=batch_size, seed=seed, test_freq=test_freq, trial="", dataset=dataset)

    else:
        # bias-variance code 
        permute_index = np.split(np.random.permutation(len(train_set)), num_trial)
        for trial in range(num_trial):
            
            trainsubset = get_subsample_dataset(train_set, permute_index[trial])
            
            # Train the ensemble networks
            train(model=model, train_set=trainsubset, valid_set=valid_set, test_set=test_set, save=save,
                n_epochs=n_epochs, batch_size=batch_size, seed=seed, test_freq=test_freq, trial=trial, dataset=dataset)
    
    print('Done!')

# [W] Extension of configure_and_train method for ensemble training 
def configure_and_train_denseNet_ensemble(data, save, depth=100, growth_rate=12, efficient=False, valid_size=5000,
         n_epochs=300, batch_size=64, seed=None, num_networks=4, test_freq=1, dataset = "cifar10", num_trial=1, 
         resume=False):
    
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.
    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)
        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)
        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """
    # Print and save arguments given to this function 
    print_and_save_arguments(locals(),save)

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    train_set, test_set, small_inputs, num_classes, grayscale = get_dataset(dataset, data)
    
    if valid_size:
        print("validation dataset is not defined, define it")
        return
    else:
        valid_set = None
        print("Validating on the test set")

    # Create ensemble networks
    ensemble_networks = []
    for _ in range(num_networks):
        ensemble_networks.append(
            DenseNet(
                growth_rate=growth_rate,
                block_config=block_config,
                num_classes=num_classes,
                small_inputs=small_inputs,
                efficient=efficient,
                grayscale=grayscale,)
        )        

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)
    
    
    if num_trial == 1:
        
        # Train the ensemble networks as usual
        train_ensemble(models=ensemble_networks, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save,
            n_epochs=n_epochs, batch_size=batch_size, seed=seed, test_freq=test_freq, trial="")
    else:
        # bias-variance code 
        permute_index = np.split(np.random.permutation(len(train_set)), num_trial)
        for trial in range(num_trial):
            
            trainsubset = get_subsample_dataset(train_set, permute_index[trial])
            
            # Train the ensemble networks
            train_ensemble(models=ensemble_networks, train_set=trainsubset, valid_set=valid_set, test_set=test_set, save=save,
                n_epochs=n_epochs, batch_size=batch_size, seed=seed, test_freq=test_freq, trial=trial)
    print('Done!')