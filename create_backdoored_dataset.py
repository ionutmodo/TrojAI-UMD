import os
import ast
import shutil
import pandas as pd
import multiprocessing as mp
from tools.logistics import get_project_root_path
from tools.data import create_backdoored_dataset

IMAGES_PER_CLASS = 50  # 50 for round 1
# IMAGES_PER_CLASS = 10  # 50 for round 2


def create_dataset_multiprocessing(dict_params):
    images_count_per_class = dict_params['images_per_class']
    num_classes = dict_params['num_classes']
    params_method = dict_params['params_method']

    path_data_backd = params_method['dir_backdoored_data']
    if os.path.isdir(path_data_backd):
        files_count_in_backd_dataset = len(os.listdir(path_data_backd))
        if files_count_in_backd_dataset == (num_classes * images_count_per_class + 1): # +1 because of csv data
            return
        shutil.rmtree(path_data_backd)

    print(f'started {path_data_backd}')
    create_backdoored_dataset(**params_method)
    print(f'done {path_data_backd}')


def main():
    path_trigger = 'square'
    path_root_project = get_project_root_path()
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-dataset-train')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-holdout-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')
    metadata = pd.read_csv(path_metadata)

    if 'train' in os.path.basename(path_root): # append 'models' for training dataset
        path_root = os.path.join(path_root, 'models')

    mp_mapping_params = []
    list_trigger_sizes = [15, 20, 25, 40, 45, 50]

    for _, row in metadata.iterrows():
        model_name = row['model_name']
        num_classes = row['number_classes']

        trigger_color = row['trigger_color']
        triggered_classes = ast.literal_eval(row['triggered_classes'].replace(' ', ', '))
        trigger_target_class = row['trigger_target_class']
        trigger_target_class = int(trigger_target_class) if trigger_target_class != 'None' else 0  # default class

        if len(triggered_classes) == 0:
            triggered_classes = list(range(num_classes))

        if trigger_color == 'None':
            trigger_color = (0, 0, 0)  # default color
        else:  # reversed because round 1 uses BGR
            trigger_color = tuple(reversed(ast.literal_eval(row['trigger_color'].replace(' ', ', '))))

        ###############################################################################################################

        path_model = os.path.join(path_root, model_name)
        if os.path.isdir(path_model):
            path_data_clean = os.path.join(path_model, 'example_data')
            # for a path_data_clean, generate a path_data_backd with a for a specific size for square trigger
            for p_trigger_size in list_trigger_sizes:
                exp_desc = f'custom-square-size-{p_trigger_size}_backd-original-color_clean-black-color'
                path_data_backd = os.path.join(path_model, f'backdoored_data_{exp_desc}')

                params_method = dict(
                    dir_clean_data=path_data_clean,
                    dir_backdoored_data=path_data_backd,
                    filename_trigger=path_trigger,
                    triggered_fraction=1.0,  # only polygon trigger
                    triggered_classes=triggered_classes,
                    trigger_target_class=trigger_target_class,
                    trigger_color=trigger_color,
                    trigger_size=p_trigger_size,
                    p_trigger=1.0,
                    keep_original=False)

                mapping_param_dict = dict(
                    params_method=params_method,
                    num_classes=num_classes,
                    images_per_class=IMAGES_PER_CLASS
                )
                mp_mapping_params.append(mapping_param_dict)

    with mp.Pool(processes=mp.cpu_count()-4) as pool:
        pool.map(create_dataset_multiprocessing, mp_mapping_params)


if __name__ == '__main__':
    main()
    print('script ended')
