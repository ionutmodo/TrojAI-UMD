import socket
import sys

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

    sys.stdout.flush()
    ics = LayerwiseClassifiers(output_params, architecture_params).to(device)
    ics.set_model(model)

    optimizer, scheduler = af.get_optimizer(ics, lr_params, stepsize_params, optimizer='adam')

    print(f'Training SDN version for model {os.path.basename(model_root_path)}')
    sys.stdout.flush()
    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    test_proc = int(dataset.test_ratio * 100)
    train_proc = 100 - test_proc
    bs = dataset.batch_size
    arcs.save_model(ics, params, model_root_path, f'ics_train{train_proc}_test{test_proc}_bs{bs}', epoch=-1)


def main():
    sys.stdout.flush()
    random_seed = af.get_random_seed()
    af.set_random_seeds()

    device = af.get_pytorch_device()
    # device = 'cpu'

    hostname = socket.gethostname()
    hostname = 'openlab' if hostname.startswith('openlab') else hostname

    print('Random Seed: {}'.format(random_seed))
    print(f'Running on machine "{hostname}"')

    hostname_root_dict = {
        'ubuntu20': '/mnt/storage/Cloud/MEGA/TrojAI',  # the name of ionut's machine
        'openlab': '/fs/sdsatumd/ionmodo/TrojAI'
    }
    root_path = os.path.join(hostname_root_dict[hostname], 'TrojAI-data', 'round1-holdout-dataset')

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    # get folder names of DenseNet121 models
    model_ids = metadata[metadata['model_architecture'] == 'densenet121']['model_name'].tolist()

    num_classes = 5 # read this from metadata
    batch_size = 25
    test_ratio = 0
    sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks

    for _id in [9]:
        model_id = f'id-{_id:08d}'
        model_root = os.path.join(root_path, model_id)  # the folder where model, example_data and ground_truth.csv are stored
        print(f'Working directory: {model_root}')
        dataset, model_label, model = read_model_directory(model_root, num_classes, batch_size, test_ratio, sdn_type, device)

        train_trojai_sdn(dataset, model, model_root, device)
    print('script ended')


if __name__ == "__main__":
    main()
