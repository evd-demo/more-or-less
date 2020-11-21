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
import torch.nn.functional as F

def get_ensemble_average(models, input, num_classes=10):
    
    num_classes = models[0].num_classes
    output = torch.zeros([input.shape[0],num_classes])
    
    if torch.cuda.is_available():
        output = output.cuda()

    for m in models:
        output += m(input)
    return output/len(models)*1.0

#########################################################################################################
def kl_div_cal(P, Q):
    return (P * (P / Q).log()).sum()

def compute_log_output_kl(net, testloader, ensemble=False):
    
    test_size = 10000
    num_classes = 10
    
    if ensemble:
        for n in net:
            n.eval()
    else:
        net.eval()
    
    total = 0
    outputs_log_total = torch.Tensor(test_size, num_classes).zero_()
    
    debug = True
    if torch.cuda.is_available():
        outputs_log_total = outputs_log_total.cuda()
        debug = False
   
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            if ensemble:
                outputs = get_ensemble_average(net,inputs)
            else:
                outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)
            outputs_log_total[total:(total + targets.size(0)), :] = outputs.log()
            total += targets.size(0)
            if debug:
                break
    return outputs_log_total.cpu()


def compute_normalization_kl(outputs_log_avg):
    test_size = 10000
    num_classes = 10

    outputs_norm = torch.Tensor(test_size, num_classes).zero_()
    if torch.cuda.is_available():
        outputs_norm = outputs_norm.cuda()
        outputs_log_avg = outputs_log_avg.cuda()

    for idx in range(test_size):
        for idx_y in range(num_classes):
            outputs_norm[idx, idx_y] = torch.exp(outputs_log_avg[idx, idx_y])
    for idx in range(test_size):
        y_total = 0.0
        for idx_y in range(num_classes):
            y_total += outputs_norm[idx, idx_y]
        outputs_norm[idx, :] /= (y_total * 1.0)
    return outputs_norm.cpu()

def compute_bias_variance_ce(networks, testloader, num_classes=10, ensemble=False, ensemble_size=4):

    """
        INPUT: networks -- contain network to be evaluated together for bias_variance calculation
               test_loader -- you know what it is
               num_classes 10 for C10, 100 for C100 etc.
        OUTPUT: Returns the bias an variance as explained here: https://arxiv.org/pdf/2002.11328.pdf
        (https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff)
    """

    ##########################################
    # Initialize variables to keep track
    ##########################################

    num_trials = len(networks)
    test_size = 10000 #  hard-coded
    
    outputs_log_avg = torch.Tensor(test_size, num_classes).zero_()

    # bias-variance
    bias2 = torch.zeros(num_trials)
    variance = torch.zeros(num_trials)
    nll_loss = nn.NLLLoss(reduction='sum')
        
    # test loss
    test_loss = torch.zeros(num_trials)
    test_acc = torch.zeros(num_trials)
    correct = torch.zeros(num_trials)
    criterion = nn.CrossEntropyLoss().cuda()

    # shared
    total = torch.zeros(num_trials)
     
    trial = 0
    ##########################################
    # Compute the log-output average
    ##########################################

    for net in networks:

        if ensemble:
            print('e ', )
            for i in range(ensemble_size):
                net[i].eval 
                if torch.cuda.is_available():
                    net[i] = net[i].cuda()
        else:
            net.eval()
            if torch.cuda.is_available():
                    net = net.cuda()
        
        outputs_log_avg += compute_log_output_kl(net, testloader, ensemble)
    
    ##########################################
    # Normalization
    ##########################################

    outputs_norm = compute_normalization_kl(outputs_log_avg)

    ##########################################
    # second pass to computer bias variance
    ##########################################
    debug=True
    for net in networks:

        if ensemble:
            for i in range(ensemble_size):
                net[i].eval 
                if torch.cuda.is_available():
                    net[i] = net[i].cuda()
        else:
            net.eval()
            if torch.cuda.is_available():
                    net = net.cuda()
                    debug=False
        
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(testloader):
                print(targets.shape)
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                if ensemble:
                    outputs = get_ensemble_average(net, inputs)
                else:
                    outputs = net(inputs)
                
                outputs = F.softmax(outputs,dim=1)
                outputs = outputs.cpu()

                ##########################################
                # Compute and track the test loss
                ##########################################
                loss = criterion(outputs, targets)
                test_loss[trial] += loss.item() * outputs.numel()
                _, predicted = outputs.max(1)
                correct[trial] += predicted.eq(targets).sum().item()
                
                ##########################################
                # Compute and track bias and the variance
                ##########################################
                total_ = int(total[trial].item())

                bias2[trial] += nll_loss(outputs_norm[total_:total_ + targets.size(0), :].log(), targets)
                for idx in range(len(inputs)):
                    variance_idx = kl_div_cal(outputs_norm[total_ + idx], outputs[idx])
                    if variance_idx > -0.0001:
                        variance[trial] += variance_idx

                total[trial] += targets.size(0)
                if debug:
                    break
                

        # test loss
        test_loss[trial] = test_loss[trial] / total[trial]
        test_acc[trial] = 100. * correct[trial] / total[trial]

        # bias variance 
        bias2[trial] /= total[trial]
        variance[trial] /= total[trial]

        
        print("trial: {0}, var: {1}, bias2: {2}".format(trial, variance, bias2))
        trial+=1
        
    
    return bias2.mean(), variance.mean()



#########################################################################################################

def compute_bias_variance_mse(networks, testloader, num_classes=10, ensemble=False, ensemble_size=4):

    """
        INPUT: networks -- contain network to be evaluated together for bias_variance calculation
               test_loader -- you know what it is
               num_classes 10 for C10, 100 for C100 etc.
        OUTPUT: Returns the bias an variance as explained here: https://arxiv.org/pdf/2002.11328.pdf
        (https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff)
    """

    num_trials = len(networks)
    test_size = 10000 # 
    
    outputs_sum = torch.Tensor(test_size, num_classes).zero_()
    outputs_sumnormsquared = torch.Tensor(test_size).zero_()

    # bias-variance
    bias2 = torch.zeros(num_trials)
    variance = torch.zeros(num_trials)
    variance_unbias = torch.zeros(num_trials)
    bias2_unbias = torch.zeros(num_trials)
        
    #test loss
    test_loss = torch.zeros(num_trials)
    test_acc = torch.zeros(num_trials)
    correct = torch.zeros(num_trials)
    criterion = nn.MSELoss(reduction='mean').cuda()

    # shared
    total = torch.zeros(num_trials)
    
    # if torch.cuda.is_available():
    #     outputs_sum = outputs_sum.cuda()
    #     outputs_sumnormsquared = outputs_sumnormsquared.cuda()
    #     variance = variance.cuda()
    #     total = total.cuda()
    #     test_loss = test_loss.cuda()
    #     correct = correct.cuda()
    #     criterion = criterion.cuda()
    
    trial = 0

    for net in networks:

        if ensemble:
            print('e ', )
            for i in range(ensemble_size):
                net[i].eval 
                if torch.cuda.is_available():
                    net[i] = net[i].cuda()
        else:
            net.eval()
            if torch.cuda.is_available():
                    net = net.cuda()
        
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(testloader):
                
                targets_onehot = torch.FloatTensor(targets.size(0), num_classes)
                # print(epoch)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                if ensemble:
                    outputs = get_ensemble_average(net, inputs)
                else:
                    outputs = net(inputs)
                
                outputs = F.softmax(outputs,dim=1)
                outputs = outputs.cpu()

                targets_onehot.zero_()
                targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
                # test loss
                loss = criterion(outputs, targets_onehot)
                test_loss[trial] += loss.item() * outputs.numel()
                _, predicted = outputs.max(1)
                correct[trial] += predicted.eq(targets).sum().item()
                
                # bias-variance
                total_ = int(total[trial].item())
                outputs_sum[total_:(total_ + targets.size(0)), :] += outputs
                outputs_sumnormsquared[total_:total_ + targets.size(0)] += outputs.norm(dim=1) ** 2.0

                bias2[trial] += (outputs_sum[total_:total_ + targets.size(0), :] / (trial + 1) - targets_onehot).norm() ** 2.0
                variance[trial] += outputs_sumnormsquared[total_:total_ + targets.size(0)].sum()/(trial + 1) - (outputs_sum[total_:total_ + targets.size(0), :]/(trial + 1)).norm() ** 2.0
                total[trial] += targets.size(0)
                

        # test loss
        test_loss[trial] = test_loss[trial] / total[trial]
        test_acc[trial] = 100. * correct[trial] / total[trial]

        # bias variance 
        bias2[trial] /= total[trial]
        variance[trial] /= total[trial]

        variance_unbias[trial] = variance[trial] * num_trials / (num_trials - 1.0)
        bias2_unbias[trial] = (test_loss.sum() / (trial + 1)) - variance_unbias[trial]
        
        print("trial: {0}, var: {1}, bias2: {2}".format(trial, variance_unbias, bias2_unbias))
        trial+=1
        
    
    return bias2_unbias.mean(), variance_unbias.mean()

def load_pretrained_models(directory, depths, growth_rates, num_trials=2, num_classes=10, ensemble_size=4, dataset="cifar10", vary="d"):
    """
        Return pretrained model stored in specified directory
        OUTPUT -- OrderedDict, KEYS: gr|d|e, value: array of size num_trials; each 
                    position stores the pretrained model
    """
    result = {}
    result['single'] = OrderedDict()
    result['vertical'] = OrderedDict()
    result['horizontal'] = OrderedDict()

    for growth_rate, depth in list(itertools.product(growth_rates, depths)):
        
        d = "gr_" + str(growth_rate) + "_d_" + str(depth) + "_e_" + str(ensemble_size) + "_" + str(dataset)

        k = "|".join([str(growth_rate), str(depth), str(ensemble_size)])

        result['single'][k] = []
        result['vertical'][k] = []
        result['horizontal'][k] = []
        
        for i in range(num_trials):

            # load single network
            os.chdir(os.path.join(directory, d+"_vd")) 
            model = DenseNet(
                growth_rate=growth_rate,
                block_config=[(depth - 4) // 6 for _ in range(3)],
                num_classes=num_classes,
                small_inputs=True,
                efficient=False,
            )
            state_dict = torch.load(str(i)+"single_model.dat", map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            result['single'][k].append(model)

            # vertical ensembles
            os.chdir(os.path.join(directory, d+"_vd")) 
            ensemble_depth, _ = setup.get_ensemble_depth(depth=depth, growth_rate=growth_rate, ensemble_size=ensemble_size)
            
            vertical_ensembles = []
            for e in range(ensemble_size):
                vertical_ensembles.append(
                    DenseNet(
                        growth_rate=growth_rate,
                        block_config=[(ensemble_depth - 4) // 6 for _ in range(3)],
                        num_classes=num_classes,
                        small_inputs=True,
                        efficient=False,
                    ))
                state_dict = torch.load(str(e) + str(i) + "_model.dat", map_location=torch.device('cpu'))
                vertical_ensembles[e].load_state_dict(state_dict) 
            result['vertical'][k].append(vertical_ensembles)
            
            # horizontal ensembles
            os.chdir(os.path.join(directory, d+"_vgr"))
            ensemble_growth_rate, _ = setup.get_ensemble_growth_rate(depth=depth, growth_rate=growth_rate, ensemble_size=ensemble_size)
            horizontal_ensembles = []

            for e in range(ensemble_size):
                horizontal_ensembles.append(
                    DenseNet(
                    growth_rate=ensemble_growth_rate,
                    block_config=[(depth - 4) // 6 for _ in range(3)],
                    num_classes=num_classes,
                    small_inputs=True,
                    efficient=False,
                ))
                state_dict = torch.load(str(e) + str(i) + "_model.dat", map_location=torch.device('cpu'))
                horizontal_ensembles[e].load_state_dict(state_dict)
            result['horizontal'][k].append(horizontal_ensembles)
            
        os.chdir(directory)
    return result