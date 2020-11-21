import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from utils.imagenet_prep import *
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def convert_data(d, img_size=32, mirror=False):
    
    x = d['data']
    y = d['labels']
    
    x = x/np.float32(255)
    
    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    
    if mirror:
        # create mirrored images
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train, X_train_flip), axis=0)
        Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)


    return dict(
        X=X_train,
        Y=Y_train)

def get_imagenet_downsampled(train=False, val=False):

    assert((train and val) == False)
    
    directory = "./data/imagenet-downsampled/"

    ###################
    #   Training data
    ###################
    
    if train: 
        
        
        for i in range(1,11):
            
            print(i)
            #################################################
            #   Load, convert, and append every chunk of data
            #################################################
            data = np.load(directory + "train/train_data_batch_" + str(i) + ".npz")
            converted_data = convert_data(data)

            if i ==1:
                data_X = converted_data['X']
                data_Y = converted_data['Y']
            else:
                data_X = np.concatenate([data_X, converted_data['X']])
                data_Y = np.concatenate([data_Y, converted_data['Y']])

        return TensorDataset(torch.Tensor(data_X), torch.Tensor(data_Y).type(torch.LongTensor))
    
    ###################
    #   validation data
    ###################
    
    if val:
            
        #################################################
        #   Load, convert, and append every chunk of data
        #################################################
        data = np.load(directory + "val/val_data.npz")
        converted_data = convert_data(data)

        
        #####################################################
        #   Convert to torch data set
        #####################################################

        return TensorDataset(torch.Tensor(converted_data['X']), torch.Tensor(converted_data['Y']).type(torch.LongTensor))

def get_dataset(dataset, directory):

    print("dataset:", dataset)
    data=directory

    ##########################################################################################################
    if dataset not in ['cifar10', 'cifar100','imagenet', 'svhn', 'imagenet32']:
        print("data set not found")
        return
    
    small_inputs = True
    grayscale = False
    # CIFAR
    if dataset == "cifar10" or dataset == "cifar100": 
        
        mean = [0.5071, 0.4867, 0.4408]
        stdv = [0.2675, 0.2565, 0.2761]
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        if dataset == "cifar10":
            data_function = datasets.CIFAR10
            num_classes = 10
        elif  dataset == "cifar100":
            data_function = datasets.CIFAR100
            num_classes = 100

        train_set = data_function(data, train=True, transform=train_transforms, download=True)
        test_set = data_function(data, train=False, transform=test_transforms, download=True)

    ##########################################################################################################
    # SVHN
    if dataset == "svhn": 
        
        mean = 0.1307
        stdv = 0.3081
        train_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        
        test_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        data_function = datasets.SVHN
        num_classes = 10
        grayscale = True
        
        train_set = data_function(data, split='train', transform=train_transforms, download=True)
        test_set = data_function(data, split='test', transform=test_transforms, download=True)
    
    ##########################################################################################################
    # ImageNet
    if dataset == "imagenet":
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),

        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        num_classes = 200
        
        data = "./data/tiny-imagenet-200"
        create_val_img_folder(data)
        train_dir = os.path.join(data, 'train')
        test_dir = os.path.join(data, 'val', 'images') # testing on the validation set as the test set is not labelled in tinyimagenet

        train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
        test_set = datasets.ImageFolder(test_dir, transform=test_transforms)
        small_inputs = False
    ##########################################################################################################

    ##########################################################################################################
    # ImageNet32
    if dataset == "imagenet32":
        
        num_classes = 1000
        
        train_set = get_imagenet_downsampled(train=True)
        test_set = get_imagenet_downsampled(val=True)

        small_inputs = True
    ##########################################################################################################


    return train_set, test_set, small_inputs, num_classes, grayscale