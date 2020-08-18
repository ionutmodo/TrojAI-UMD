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
    mlp_architecture_param = [2, 2] # use [2] if it takes too much time
    # think about simplifying ICs architecture
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

    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    batch_size = 25
    test_ratio = 0
    sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks

    clean_model_ids = [4, 7, 25, 27, 40]
    backdoored_model_ids = [2, 9, 13, 24, 26]
    already_trained_model_ids = clean_model_ids + backdoored_model_ids

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        model_architecture = row['model_architecture']
        num_classes = row['number_classes']  # read this from metadata

        if model_architecture == 'densenet121':
            model_id = int(model_name[3:])
            if model_id not in already_trained_model_ids:
                model_root = os.path.join(root_path, model_name) # the folder where model, example_data and ground_truth.csv are stored
                print(f'Training SDN for model {model_root}')
                dataset, model_label, model = read_model_directory(model_root, num_classes, batch_size, test_ratio, sdn_type, device)
                train_trojai_sdn(dataset, model, model_root, device)
    print('script ended')


if __name__ == "__main__":
    main()
