import os
import shutil
import pandas as pd
from tools.logistics import *
from architectures.SDNs.SDNConfig import SDNConfig


def main():
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-dataset-train')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-train-dataset')

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    if 'train' in os.path.basename(root_path) and 'round1' in os.path.basename(root_path): # append 'models' for training dataset
        root_path = os.path.join(root_path, 'models')

    sdn_type, architecture_to_train = SDNConfig.ResNet50, 'resnet50'
    # sdn_type, architecture_to_train = SDNConfig.Inception3, 'inceptionv3'
    # sdn_type, architecture_to_train = SDNConfig.DenseNet_attach_to_DenseBlocks, 'densenet121'

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        model_architecture = row['model_architecture']
        model_id_int = int(model_name[3:])

        if model_architecture == architecture_to_train and model_id_int > 799:
            ics_folder = os.path.join(root_path, model_name, 'ics_train100_test0_bs25')
            if os.path.isdir(ics_folder):
                shutil.rmtree(ics_folder)
                print(f'deleted {ics_folder}')


if __name__ == '__main__':
    main()

# last training resnet: id-00000799
