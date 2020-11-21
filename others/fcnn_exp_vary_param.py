#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Inputs: 
    - type of network architecture
    - number of parameters
    - big network
    - config file parameters

Outputs:
    - experiment object
    - result objects
    - logs
"""

import os
import time
import argparse
from pprint import pprint
from datetime import datetime
from os.path import join as JP
# from collections import OrderedDict as OD
from beautifultable import BeautifulTable as BT

from models import FCNN
from utils.data import create_data_loaders 
from utils.helpers import (
    parse_yaml, print_current_config, 
    ensure_directories, path_check,
    count_parameters, solve_quadratic_equation
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Naming helpers
net_ = lambda n: 'net_{}'.format(n)
opt_ = lambda n: 'opt_{}'.format(n)


""" CONFIG """

CUDA = torch.cuda.is_available()
N_GPU = torch.cuda.device_count()
DEVICE = 'cuda' if CUDA else 'cpu'
WORKERS = torch.multiprocessing.cpu_count()
WORKERS = 1
print_current_config(CUDA, N_GPU, DEVICE, WORKERS)


""" EXPERIMENT DEFINITION """

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to train on')
parser.add_argument("--grayscale", default=True, action="store_true" , help="Grayscale input data")
parser.add_argument('--network', default='fcnn', type=str, help='network type')
parser.add_argument('--num_param', default=1024, type=float, help='number of parameters')
parser.add_argument('--num_layers', default=-1, type=int, help='number of layers')
parser.add_argument('--width', default=-1, type=int, help='width of all hidden layers')
parser.add_argument('--ens_size', default=4, type=int, help='ensemble size')
parser.add_argument('--exp_suffix', default="", type=str, help='experiment suffix')
args = parser.parse_args()
print(args)
# args = dict(network='fcnn', big='fcnn_L', small='fcnn_S', dataset='svhn', grayscale=True)
# from types import SimpleNamespace
# args = SimpleNamespace(**args)

config = parse_yaml('config.yaml')
paths = config['paths']
prmts = config['hyperparameters'][args.dataset][args.network]

SAVE_EVERY = config['settings']['save_every'] 
LOWEST_ACC = config['settings']['lowest_accy'] 
START_EPOCH = config['settings']['start_epoch'] 

BATCH_SIZE = prmts['batch_size']  
MILESTONES = prmts['milestones']  
TOTAL_EPOCHS = prmts['total_epochs']
LR_RED_SIZE = prmts['lr_red_size']

""" MODELS """

from utils.helpers import get_ensemble_size
from models.ensemble import UniformEnsemble as UEnsemble
   
# Single network -- initializing
print("Building the single network")

args.num_param = args.num_param*1000000

if args.width == -1:
    hid_dim_single = solve_quadratic_equation(args.num_layers -1, (32*32)+10, -1*args.num_param)
    num_layers_single = args.num_layers

if args.num_layers == -1:
    num_layers_single = int( ( ((args.num_param/args.width) - ((32*32) + 10)) / args.width ) + 1)
    hid_dim_single = args.width



single_network = FCNN(name="single", 
                num_layers=num_layers_single, 
                hid_dim=hid_dim_single, dataset=args.dataset)

optimizer = optim.SGD(
    params=single_network.parameters(), lr=prmts['learning_rate'],
    momentum=prmts['momentum'], weight_decay=prmts['weight_decay'])

single = dict(name=single_network.name, net=single_network, optimizer=optimizer)

print("single")
print(num_layers_single, hid_dim_single)
print(count_parameters(single['net'])/1e6)



# ensemble networks -- initializing
print("Building the ensemble networks")

if args.width == -1:
    hid_dim_ensemble = solve_quadratic_equation(args.num_layers -1, (32*32)+10, -1*args.num_param/args.ens_size)
    num_layers_ensemble = args.num_layers

if args.num_layers == -1:
    num_layers_ensemble = int( ( ((args.num_param/(args.width*args.ens_size)) - ((32*32) + 10)) / args.width ) + 1)
    hid_dim_ensemble = args.width


opt = {'optimizer': optim.SGD, 'lr':prmts['learning_rate'], 'momentum':prmts['momentum'], 'weight_decay':prmts['weight_decay']}
ensemble_function = "FCNN(name=\'ensemble\', num_layers =" +str(num_layers_ensemble)+ ", hid_dim=" +str(hid_dim_ensemble)+ ", dataset='" +str(args.dataset)+"')"
ensemble = UEnsemble(network=ensemble_function, optimizer=opt, size=args.ens_size)

table = BT()
table.append_row([single['name'], '{} M.'.format(count_parameters(single['net'])/1e6)])
table.append_row([ensemble.networks['net_1'].name, '{} M.'.format(count_parameters(ensemble.networks['net_1'])/1e6)])
table.append_row(['Epochs', TOTAL_EPOCHS])
table.append_row(['Batch Size', BATCH_SIZE])
table.append_row(['Milestones', MILESTONES])
table.append_row(['Ensemble Size', args.ens_size])
print('Training Settings')
print(table)

print("single")
print(num_layers_ensemble, hid_dim_ensemble)


# 
exit()

""" TRAINING """

from utils.results import Results
from utils.results import Experiment
from utils.results import EnsembleResults as EResults
from utils.helpers import timeit, count_parameters
from train import train, test
from train import train_ensemble, test_ensemble
from train import lr_schedule, lr_schedule_ensemble, save_models

single_result = Results(name=single_network.name)
ensemble_results = EResults(size=ensemble.size, name=ensemble.name)
experiment_name = str(args.num_param*1.0/1000000.0) + "M_" + str(args.ens_size)
experiment = Experiment(experiment_name, single_result, ensemble_results)
print('\nEXPERIMENT: ', experiment_name)

# ==> Creating and ensuring directories for this network type <==
for k,p in paths.items():
    if k in ['logs', 'experiments', 'tables']:
        ensure_directories(JP(paths[k], args.dataset, args.network+args.exp_suffix))
    else:
        ensure_directories(JP(paths[k], args.dataset, args.network+args.exp_suffix, experiment_name))
print('\nPaths:')
pprint(paths)

BEST_ACC = BEST_ACC_ENSEMBLE = 0
criterion = nn.CrossEntropyLoss().to(DEVICE)


""" DATA """

GRAYSCALE = args.grayscale
train_loader, valid_loader = create_data_loaders(args.dataset, BATCH_SIZE, GRAYSCALE, WORKERS)


@timeit
def run_epoch(
    epoch, net, optimizer, criterion, trainloader, testloader):
    
    global DEVICE
    global BEST_ACC
    global LOWEST_ACC
    global MILESTONES
    global SAVE_EVERY

    print('\nSingle :: Epoch ', epoch)
    if epoch in MILESTONES:
        lr_schedule(optimizer)
    
    tr_loss, tr_accy = train(net, optimizer, criterion, trainloader, DEVICE)
    va_loss, va_accy = test(net, optimizer, criterion, testloader, DEVICE)
    return tr_loss, tr_accy, va_loss, va_accy

@timeit
def run_epoch_ensemble(
    epoch, ensemble, criterion, trainloader, testloader):
    
    global DEVICE
    global BEST_ACC
    global LOWEST_ACC
    global MILESTONES
    global SAVE_EVERY

    print('\nEnsemble :: Epoch ', epoch)
    if epoch in MILESTONES:
        lr_schedule_ensemble(ensemble)
    
    tr_losses, tr_accies = train_ensemble(ensemble, criterion, trainloader, DEVICE)
    va_losses, va_accies = test_ensemble(ensemble, criterion, testloader, DEVICE)
    return tr_losses, tr_accies, va_losses, va_accies


""" Traning Loop for Ensemble """

print('\n\n[INFO]: Starting Training')
print('=========================')

print('Epoch 0:')

print('Single: ')
va_loss, va_accy = test(single['net'], optimizer, criterion, valid_loader, DEVICE)
print('Ensemble: ')
va_loss, va_accy = test_ensemble(ensemble, criterion, valid_loader, DEVICE)

start_epoch = time.time()
for epoch in range(1+START_EPOCH, 1+TOTAL_EPOCHS):

    # SINGLE - TRAIN EPOCH 
    start_single = time.time()
    tr_loss, tr_accy, va_loss, va_accy = run_epoch(
        epoch, single['net'], single['optimizer'], criterion, train_loader, valid_loader)
    delta_single = time.time() - start_single

    # ENSEMBLE - TRAIN EPOCH
    start_ensemble = time.time()
    tr_losses, tr_accies, va_losses, va_accies = run_epoch_ensemble(
        epoch, ensemble, criterion, train_loader, valid_loader)
    delta_ensemble = time.time() - start_ensemble

    # SINGLE - Save if best model so far
    if va_accy > LOWEST_ACC:
        if va_accy > BEST_ACC:
            save_models(epoch, single['net'], single['optimizer'], va_accy, JP(
                paths['checkpoints'], args.dataset, args.network + args.exp_suffix, experiment_name,
                 single['net'].name + config['extensions']['checkpoints']))
            BEST_ACC = va_accy
    
    # ENSEMBLE - Save if best model so far
    if va_accies['ensemble'] > LOWEST_ACC:
        if va_accies['ensemble'] > BEST_ACC_ENSEMBLE:
            for n in range(1,1+ensemble.size):
                save_models(
                    epoch, ensemble.networks[net_(n)], ensemble.optimizers['opt_{}'.format(n)], 
                    va_accy, JP(
                        paths['checkpoints'], args.dataset, args.network+args.exp_suffix, experiment_name, 
                        net_(n) + config['extensions']['checkpoints']))
            BEST_ACC_ENSEMBLE = va_accies['ensemble']
    
    # Update Results Objects and Experiment
    single_result.update((tr_loss, tr_accy, va_loss, va_accy, epoch, delta_single))          
    ensemble_results.update((tr_losses, tr_accies, va_losses, va_accies,  epoch, delta_ensemble))

    # Save Results
    if epoch % SAVE_EVERY == 0:
        single_result.save_pickle(
            JP(paths['objects'], args.dataset, args.network+args.exp_suffix, experiment_name, single['net'].name + config['extensions']['results']))
        ensemble_results.save_pickle(
            JP(paths['objects'], args.dataset, args.network+args.exp_suffix, experiment_name, net_(n) + config['extensions']['results']))


""" Storing remaining and finishing """

print('[INFO]: Adding training result objects to ', experiment_name)
experiment.data['single'] = single_result
experiment.data['ensemble'] = ensemble_results
print('[INFO]: Adding trianed networks to ', experiment_name)
experiment.models['single'] = single_network.name
experiment.models['ensemble'] = ensemble.name
print('[INFO]: Saving Experiment ')
experiment.save(JP(paths['experiments'], args.dataset, args.network+args.exp_suffix, experiment_name + config['extensions']['experiments']))
print('[INFO]: Quiting... ')
exit()