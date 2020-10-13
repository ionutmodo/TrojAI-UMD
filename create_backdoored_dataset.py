import os
import ast
import shutil
import socket
import pandas as pd
import multiprocessing as mp
from tools.logistics import get_project_root_path
from tools.data import create_backdoored_dataset

IMAGES_PER_CLASS = 100  # 50 for round 1
# IMAGES_PER_CLASS = 10  # 50 for round 2


def create_dataset_multiprocessing(dict_params):
    images_count_per_class = dict_params['images_per_class']
    num_classes = dict_params['num_classes']
    params_method = dict_params['params_method']

    path_data_backd = params_method['dir_backdoored_data']
    items = path_data_backd.split(os.path.sep)
    short_path = os.path.join(items[-3], items[-2], items[-1])

    if os.path.isdir(path_data_backd):
        files_count_in_backd_dataset = len(os.listdir(path_data_backd))
        if files_count_in_backd_dataset == (num_classes * images_count_per_class + 1): # +1 because of csv data
            print(f'already complete ({files_count_in_backd_dataset}) {short_path}')
            return
        shutil.rmtree(path_data_backd)
        print(f'deleted ({files_count_in_backd_dataset}) {path_data_backd}')

    create_backdoored_dataset(**params_method)
    print(f'done {short_path}')


def main():
    path_root_project = get_project_root_path()
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-dataset-train')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-holdout-dataset')
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round2-train-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')
    metadata = pd.read_csv(path_metadata)

    if 'train' in os.path.basename(path_root) and 'round1' in os.path.basename(path_root): # append 'models' for training dataset
        path_root = os.path.join(path_root, 'models')

    mp_mapping_params = []
    # list_trigger_sizes = [15, 20, 25, 40, 45, 50]
    list_trigger_sizes = [20]
    list_filters = ['gotham', 'kelvin', 'lomo'] #'nashville'

    list_limits = {
        'openlab30.umiacs.umd.edu': (0, 275),
        'openlab31.umiacs.umd.edu': (276, 551),
        'openlab32.umiacs.umd.edu': (552, 827),
        'openlab33.umiacs.umd.edu': (828, 1103)
    }

    for _, row in metadata.iterrows():
        model_name = row['model_name']
        model_id = int(model_name[3:])
        left, right = list_limits[socket.gethostname()]
        if left <= model_id <= right:
            num_classes = row['number_classes']
            number_example_images = int(row['number_example_images'])
            trigger_color = row['trigger_color']
            triggered_classes = row['triggered_classes']

            if triggered_classes.lower() == 'none':
                triggered_classes = '[]'
            triggered_classes = ast.literal_eval(triggered_classes.replace(' ', ', '))

            # default class for clean models
            # I set trigger_target_class to 0 to pass a valid parameter to method create_backdoored_dataset
            trigger_target_class = row['trigger_target_class']
            trigger_target_class = int(trigger_target_class) if trigger_target_class.lower() != 'none' else 0

            if len(triggered_classes) == 0:
                triggered_classes = list(range(num_classes))

            if trigger_color.lower() == 'none':
                trigger_color = (0, 0, 0) # default color
            else: # do not reverse color anymore
                trigger_color = tuple(ast.literal_eval(row['trigger_color'].replace(' ', ', ')))

            ###############################################################################################################

            path_model = os.path.join(path_root, model_name)
            if os.path.isdir(path_model):
                path_data_clean = os.path.join(path_model, 'example_data')
                # for a path_data_clean, generate a path_data_backd with a for a specific size for square trigger
                # generate backdoored datasets with square trigger with specific size
                for p_trigger_size in list_trigger_sizes:
                    exp_desc = f'custom-square-size-{p_trigger_size}_backd-original-color_clean-black-color'
                    path_data_backd = os.path.join(path_model, f'backdoored_data_{exp_desc}')

                    mapping_param_dict = dict(
                        num_classes=num_classes,
                        images_per_class=number_example_images,
                        params_method=dict(
                            dir_clean_data=path_data_clean,
                            dir_backdoored_data=path_data_backd,
                            trigger_type='polygon',
                            trigger_name='square',
                            trigger_color=trigger_color,
                            trigger_size=p_trigger_size,
                            triggered_classes=triggered_classes,
                            trigger_target_class=trigger_target_class)
                    )
                    mp_mapping_params.append(mapping_param_dict)
                    # create_dataset_multiprocessing(mapping_param_dict)

                # # generate backdoored datasets with specific filter
                # for p_filter_name in list_filters:
                #     path_data_backd = os.path.join(path_model, f'backdoored_data_filter_{p_filter_name}')
                #
                #     mapping_param_dict = dict(
                #         num_classes=num_classes,
                #         images_per_class=number_example_images,
                #         params_method=dict(
                #             dir_clean_data=path_data_clean,
                #             dir_backdoored_data=path_data_backd,
                #             trigger_type='filter',
                #             trigger_name=p_filter_name,
                #             trigger_color=None,
                #             trigger_size=None,
                #             triggered_classes=triggered_classes,
                #             trigger_target_class=trigger_target_class)
                #     )
                #     mp_mapping_params.append(mapping_param_dict)
                #     # create_dataset_multiprocessing(mapping_param_dict)

    cpus = mp.cpu_count() - 4
    print(f'Creating {len(mp_mapping_params)} datasets using {cpus} CPU cores')
    with mp.Pool(processes=cpus) as pool:
        pool.map(create_dataset_multiprocessing, mp_mapping_params)


if __name__ == '__main__':
    main()
    print('script ended')
