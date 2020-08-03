import os

import numpy as np
import torch
import torch.nn as nn
import sys
import dill

import aux_funcs as af
import network_architectures as arcs
import model_funcs as mf
from SDNDenseNet121 import SDNDenseNet121

from data import ManualDataset

from MLP import LayerwiseClassifiers
# from encoder import LayerwiseAutoencoders


# from robustness.attacker import AttackerModel
# from robustness.datasets import CIFAR

# from architectures.CNNs.resnet import ResNet50
# from architectures.CNNs.wideresnet import WideResNet

def train_wrapper(model, models_path, model_name, task, device):

    print(f'training layerwise models: {models_path}, {model_name}, {task}')
    output_params = model.get_layerwise_model_params()

    mlp_num_layers = 2
    mlp_architecture_param = [2,2]
    architecture_params = (mlp_num_layers, mlp_architecture_param)

    params = {}
    params['network_type'] = 'layerwise_classifiers'
    params['output_params'] = output_params
    params['architecture_params'] = architecture_params

    # train the mlps
    epochs = 20
    lr_params = (0.001, 0.00001)
    stepsize_params = ([10], [0.1])

    ics = LayerwiseClassifiers(output_params, architecture_params).to(device)
    ics.set_model(model)

    dataset = af.get_dataset(task, batch_size=2, num_holdout=0.2)
    optimizer, scheduler = af.get_optimizer(ics, lr_params, stepsize_params, optimizer='adam')
    torch.cuda.empty_cache()
    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    arcs.save_model(ics, params, models_path, '{}_layerwise_classifiers'.format(model_name), epoch=-1)

    print('EXITING')
    sys.exit(666)

    # train the aes
    ae_num_layers = 3
    mlp_architecture_param = [0.4, 0.4, 0.4]
    architecture_params = (ae_num_layers, mlp_architecture_param)

    params = {}
    params['network_type'] = 'layerwise_autoencoders'
    params['output_params'] = output_params
    params['architecture_params'] = architecture_params

    lr_params = (0.001, 1e-9)
    stepsize_params = ( [30, 50], [0.1, 0.1])
    epochs = 60
    
    aes = LayerwiseAutoencoders(output_params, architecture_params).to(device)
    aes.set_model(model)

    dataset = af.get_dataset(task, num_holdout=0.2)
    optimizer, scheduler = af.get_optimizer(aes, lr_params, stepsize_params, optimizer='adam')
    mf.train_layerwise_autoencoders(aes, dataset, epochs, optimizer, scheduler, device)
    arcs.save_model(aes, params, models_path, '{}_layerwise_autoencoders'.format(model_name), epoch=-1)


def train_std_model(task, network):
    models_path = '/mnt/storage/Cloud/MEGA/TrojAI/code/models/'
    model_name = '{}_{}_cnn'.format(task, network)
    # model, model_params = arcs.load_model(models_path, model_name, -1)
    # model = model.to(device).eval()
    # print(model_params)
    # print(model)
    # sys.exit(666)

    model = torch.load('/mnt/storage/Cloud/MEGA/TrojAI/data/trojai-round0-dataset/id-00000005/model.pt')
    model = model.to(device).eval()
    model = SDNDenseNet121(model, input_size=(1, 3, 224, 224), num_classes=10, sdn_type=0, device=device)
    train_wrapper(model, models_path, model_name, task, device)


def train_robust(madry=True):

    if madry:
        path = './robust_models/madry'
        robust_model = af.load_madry_robust_cifar10(path, device)
        name = 'robust_madry'
    else: # trades
        path = './robust_models/trades'
        robust_model = af.load_trades_robust_cifar10(path, device)
        name = 'robust_trades'

    train_wrapper(robust_model, path, name, 'cifar10', device)


if __name__ == "__main__":
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))

    device = af.get_pytorch_device()
    # device = 'cpu'

    #tasks = ['cifar10', 'cifar100', 'tinyimagenet']
    tasks = ['cifar10']
    # networks = ['vgg16bn', 'resnet50', 'wideresnet']
    networks = ['vgg16bn']

    for task in tasks:
        for network in networks:
            train_std_model(task, network)

    # train_robust()
