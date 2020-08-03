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

from data import ManualDataset, TrojAI

from MLP import LayerwiseClassifiers

def train_wrapper(model, path, model_name, images_folder_name, task, device):
    print(f'training layerwise models: {path}, {model_name}, {task}')
    output_params = model.get_layerwise_model_params()

    mlp_num_layers = 2
    mlp_architecture_param = [2, 2]
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

    dataset = TrojAI(folder=os.path.join(path, images_folder_name), batch_size=2, device=device)
    optimizer, scheduler = af.get_optimizer(ics, lr_params, stepsize_params, optimizer='adam')
    torch.cuda.empty_cache()
    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    arcs.save_model(ics, params, path, '{}_layerwise_classifiers'.format(model_name), epoch=-1)

    # # train the aes
    # ae_num_layers = 3
    # mlp_architecture_param = [0.4, 0.4, 0.4]
    # architecture_params = (ae_num_layers, mlp_architecture_param)
    #
    # params = {}
    # params['network_type'] = 'layerwise_autoencoders'
    # params['output_params'] = output_params
    # params['architecture_params'] = architecture_params
    #
    # lr_params = (0.001, 1e-9)
    # stepsize_params = ( [30, 50], [0.1, 0.1])
    # epochs = 60
    #
    # aes = LayerwiseAutoencoders(output_params, architecture_params).to(device)
    # aes.set_model(model)
    #
    # dataset = af.get_dataset(task, num_holdout=0.2)
    # optimizer, scheduler = af.get_optimizer(aes, lr_params, stepsize_params, optimizer='adam')
    # mf.train_layerwise_autoencoders(aes, dataset, epochs, optimizer, scheduler, device)
    # arcs.save_model(aes, params, models_path, '{}_layerwise_autoencoders'.format(model_name), epoch=-1)


if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')

    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))

    device = af.get_pytorch_device()

    task = 'trojai'
    path = '/mnt/storage/Cloud/MEGA/TrojAI/data/trojai-round0-dataset/id-00000005/'
    model_name = 'model.pt'
    images_folder_name = 'example_data'

    model = torch.load(os.path.join(path, model_name)).to(device).eval()
    model = SDNDenseNet121(model, input_size=(1, 3, 224, 224), num_classes=5, sdn_type=0, device=device)
    train_wrapper(model, path, model_name, images_folder_name, task, device)
