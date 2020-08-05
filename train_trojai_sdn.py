import socket
import pandas as pd

import tools.aux_funcs as af
import tools.model_funcs as mf
import tools.network_architectures as arcs
from tools.logistics import *

from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.MLP import LayerwiseClassifiers

def train_trojai_sdn(dataset, model, model_root_path, device):
    output_params = model.get_layerwise_model_params()

    mlp_num_layers = 2
    mlp_architecture_param = [2, 2]
    architecture_params = (mlp_num_layers, mlp_architecture_param)

    params = {
        'network_type': 'layerwise_classifiers',
        'output_params': output_params,
        'architecture_params': architecture_params
    }

    # settings to train the MLPs
    epochs = 20
    lr_params = (0.001, 0.00001)
    stepsize_params = ([10], [0.1])

    ics = LayerwiseClassifiers(output_params, architecture_params).to(device)
    ics.set_model(model)

    optimizer, scheduler = af.get_optimizer(ics, lr_params, stepsize_params, optimizer='adam')
    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    arcs.save_model(ics, params, model_root_path, 'model_layerwise_classifiers', epoch=-1)


if __name__ == "__main__":
    random_seed = af.get_random_seed()
    af.set_random_seeds()

    # device = af.get_pytorch_device()
    device = 'cpu'
    hostname = socket.gethostname()

    print('Random Seed: {}'.format(random_seed))
    print(f'Running on machine "{hostname}"')

    hostname_root_dict = {
        'ubuntu20': '/mnt/storage/Cloud/MEGA', # the name of ionut's machine
        'opensub03.umiacs.umd.edu': '/fs/sdsatumd/ionmodo'
    }
    root_path = os.path.join(hostname_root_dict[hostname], 'TrojAI-data', 'round1-dataset-train')

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    # get folder names of DenseNet121 models
    model_ids = metadata[metadata['model_architecture'] == 'densenet121']['model_name'].tolist()

    task = 'trojai'
    num_classes = 5
    sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks

    for model_id in model_ids:
        model_root = os.path.join(root_path, 'models', model_id) # the folder where model, example_data and ground_truth.csv are stored
        dataset, model_label, model = read_model_directory(model_root, num_classes, sdn_type, device)
        train_trojai_sdn(dataset, model, model_root, device)
        break
    print('script ended')
