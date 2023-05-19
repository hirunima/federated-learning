#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
            transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def concat_weight_dataset(w):
    """
    Returns the concatanation of all of the weights.
    """
    # w_full = copy.deepcopy(w)
    w_full=[[] for x in range(len(w))] #torch.zeros((len(w))).to('cuda')#
    print(w_full)
    for i in range(1, len(w)):
        for key in w[i].keys():
            
            # 
            # print(w_full[i],w[i][key].view(-1).shape)
            # w_full[i]=torch.vstack((w_full[i].view(-1),w[i][key].view(-1)))
            w_full[i].append(w[i][key].view(-1))
            print(w[i][key].view(-1).shape)
            # print(len(w_full[i]))

            # w_full[i].concat()
            #[key] += w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
            # break
        w_full[i]=torch.cat(w_full[i], dim=0)
        # w_full[i]=w_full[i][1:]
        
        # print(len(w_full[i]))    
        break
    print('***************************************',len(w_full[i]))
    return w_full

def get_weight_dataset(w):
    """
    Returns the concatanation of each of the weights per key.
    """
    w_full = copy.deepcopy(w[0])
    
    for key in w_full.keys():
        w_full[key]=w[0][key].view(1,-1)
        for i in range(1, len(w)):
            w_full[key]=torch.cat([w_full[key],w[i][key].view(1,-1)], dim=0)
        print(w_full[key].shape,w[i][key].view(1,-1).shape)
            # print(len(w_full[i]))

            # w_full[i].concat()
            #[key] += w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
            # break
        # w_full[i]=torch.cat(w_full[i], dim=0)
        # w_full[i]=w_full[i][1:]
        

        # print(len(w_full[i]))    

    print('***************************************',w_full.keys())
    return w_full

def reshape_weights(w,w_org):
    """
    Returns the original resized weights per key.
    """
    
    for key in w_org.keys():
        w[key]=w[key].view(w_org[key].shape)
        print(w[key].shape)
        # for i in range(1, len(w)):
        #     w_full[key]=torch.cat([w_full[key],w[i][key].view(1,-1)], dim=0)
        # print(w_full[key].shape,w[i][key].view(1,-1).shape)
            # print(len(w_full[i]))

            # w_full[i].concat()
            #[key] += w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
            # break
        # w_full[i]=torch.cat(w_full[i], dim=0)
        # w_full[i]=w_full[i][1:]
        

        # print(len(w_full[i]))    

    print('***************************************',w.keys())
    return w

def reshape_weights(w,w_org):
    """
    Returns the original resized weights per key.
    """
    
    for key in w_org.keys():
        w[key]=w[key].view(w_org[key].shape)
        print(w[key].shape)
        # for i in range(1, len(w)):
        #     w_full[key]=torch.cat([w_full[key],w[i][key].view(1,-1)], dim=0)
        # print(w_full[key].shape,w[i][key].view(1,-1).shape)
            # print(len(w_full[i]))

            # w_full[i].concat()
            #[key] += w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
            # break
        # w_full[i]=torch.cat(w_full[i], dim=0)
        # w_full[i]=w_full[i][1:]
        

        # print(len(w_full[i]))    

    print('***************************************',w.keys())
    return w
    
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
