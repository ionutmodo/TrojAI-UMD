import os
import shutil
import pandas as pd
from tools.logistics import *
from architectures.SDNs.SDNConfig import SDNConfig


def main():
    # folder_name_to_delete = 'ics_train100_test0_bs25'
    folder_name_to_delete = 'backdoored_data_custom-square-size-20_backd-original-color_clean-black-color'

    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-dataset-train')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-train-dataset')

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    # if 'train' in os.path.basename(root_path) and 'round1' in os.path.basename(root_path): # append 'models' for training dataset
    #     root_path = os.path.join(root_path, 'models')

    count = 0
    for index, row in metadata.iterrows():
        model_name = row['model_name']
        # model_architecture = row['model_architecture']
        # model_id_int = int(model_name[3:])
        
        ics_folder = os.path.join(root_path, model_name, folder_name_to_delete)
        if os.path.isdir(ics_folder):
            shutil.rmtree(ics_folder)
            count += 1
            print(f'deleted ({count}) {ics_folder}')


if __name__ == '__main__':
    main()
