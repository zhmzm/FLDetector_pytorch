import sys

from models.Fed import FedAvg
from models.Update import LocalUpdate

sys.path.append('../')

from random import random
from models.test import test_img
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model
from torch.utils.data import DataLoader, Dataset
from utils.options import args_parser

import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import nn, autograd
import matplotlib
import os
import random
import time
import math
import heapq
import argparse
from models.add_trigger import add_trigger
from utils.defense import flame_analysis, multi_krum, get_update
from models.MaliciousUpdate import LocalMaliciousUpdate


def benign_train(model, dataset, args):
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.5)

    for images, labels in train_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()


def malicious_train(model, dataset, args):
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.5)

    for images, labels in train_loader:
        bad_data, bad_label = copy.deepcopy(
            images), copy.deepcopy(labels)
        for xx in range(len(bad_data)):
            bad_label[xx] = args.attack_label
            # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
            bad_data[xx] = add_trigger(args, bad_data[xx])
        images = torch.cat((images, bad_data), dim=0)
        labels = torch.cat((labels, bad_label))
        images, labels = images.to(args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()


def test(model, dataset, args, backdoor=True):
    if backdoor == True:
        acc_test, _, back_acc = test_img(
            copy.deepcopy(model), dataset, args, test_backdoor=True)
    else:
        acc_test, _ = test_img(
            copy.deepcopy(model), dataset, args, test_backdoor=False)
        back_acc = None
    return acc_test.item(), back_acc



def get_attacker_dataset(args):
    if args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            client_proportion = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
    elif args.dataset == "fashion_mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            client_proportion = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()

    data_list = []
    begin_pos = 0
    malicious_client_num = int(args.num_users * args.malicious)
    for i in range(begin_pos, begin_pos + malicious_client_num):
        data_list.extend(client_proportion[i])
    attacker_label = []
    for i in range(len(data_list)):
        attacker_label.append(dataset_train.targets[data_list[i]])
    attacker_label = np.array(attacker_label)
    client_dataset = []
    for i in range(len(data_list)):
        client_dataset.append(dataset_train[data_list[i]])
    mal_train_dataset, mal_val_dataset = split_dataset(client_dataset)
    return mal_train_dataset, mal_val_dataset


def split_dataset(dataset):
    num_dataset = len(dataset)
    # random
    data_distribute = np.random.permutation(num_dataset)
    malicious_dataset = []
    mal_val_dataset = []
    mal_train_dataset = []
    for i in range(num_dataset):
        malicious_dataset.append(dataset[data_distribute[i]])
        if i < num_dataset // 4:
            mal_val_dataset.append(dataset[data_distribute[i]])
        else:
            mal_train_dataset.append(dataset[data_distribute[i]])
    return mal_train_dataset, mal_val_dataset


def get_attack_layers_no_acc(model_param, args):
    mal_train_dataset, mal_val_dataset = get_attacker_dataset(args)
    return layer_analysis_no_acc(model_param, args, mal_train_dataset, mal_val_dataset)


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def cos_param(p1,p2):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    return cos(parameters_dict_to_vector_flt(p1),parameters_dict_to_vector_flt(p2))


def attacker(list_mal_client, num_mal, attack_type, dataset_train, dataset_test, dict_users, net_glob, args, idx=None):
    num_mal_temp=0
    if args.defence == 'fld':
        args.old_update = args.old_update_list[idx]
        
    if idx == None:
        idx = random.choice(list_mal_client)
    w, loss, args.attack_layers = None, None, None
    # craft attack model once
    if attack_type == "dba":
        num_dba_attacker = int(args.num_users * args.malicious)
        dba_group = int(num_dba_attacker / 4)
        idx = args.dba_sign % (4 * dba_group)
        args.dba_sign += 1
    local = LocalMaliciousUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], order=idx, dataset_test=dataset_test)
    print("client", idx, "--attack--")
    if num_mal_temp>0:
        temp_w = [w for i in range(num_mal_temp)]
        w = temp_w
    elif num_mal > 0:
        temp_w = [w for i in range(num_mal)]
        w = temp_w
    
    return w, loss, args.attack_layers
