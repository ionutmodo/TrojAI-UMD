import os, sys
import numpy as np
import pandas as pd
from scipy.stats import entropy
from datetime import datetime

from tools.logistics import *
from tools.logger import Logger


def get_trigger_type_aux_value(trigger_type, polygon_side_count, instagram_filter_type):
    if trigger_type == 'instagram':
        instagram_filter_type = instagram_filter_type.replace('FilterXForm', '').lower()
        return f'instagram-{instagram_filter_type}'
    else:
        if trigger_type == 'polygon':
            return f'{trigger_type}-{polygon_side_count}'
        else:
            return trigger_type.lower()


def get_confusion_matrix_stats(model, device, path_dataset, batch_size):
    dataset = TrojAI(folder=path_dataset, test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
    nc = dataset.num_classes

    matrix = np.zeros((nc, nc), dtype=np.int64)
    for image, label_true in dataset.train_loader:
        outputs = model(image.to(device))
        for i, out in enumerate(outputs):
            label_pred = out.unsqueeze(0).max(1)[1].item()
            matrix[label_true[i].item(), label_pred] += 1
        del outputs

    del dataset

    column_mean = matrix.mean(axis=0)
    proba = column_mean / column_mean.sum()

    uniform = np.ones_like(proba) / nc
    h = entropy(proba)
    kl = entropy(proba, uniform)

    return h / nc, kl / nc


def main():
    print(os.system('nvidia-smi'))
    # device = 'cpu'
    device = af.get_pytorch_device()

    trigger_color = (127, 127, 127)
    batch_size = 50
    if trigger_color == 'random':
        square_dataset_name = 'backdoored_data_square-30_random'
        experiment_name = f'fc_square30-random_filters_h_kl'
    else:
        square_dataset_name = 'backdoored_data_square-30'
        experiment_name = f'fc_square30-gray_filters_h_kl'

    # begin
    np.random.seed(666)
    path_root_project = get_project_root_path()
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round2-train-dataset')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round2-holdout-dataset')
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round3-train-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')
    path_report_conf_mat = os.path.join(path_root, 'conf_mat', f'{os.path.basename(path_root)}_{experiment_name}.csv')
    # path_logger = os.path.join(path_root, 'ics_svm', f'{os.path.basename(path_root)}_{experiment_name}.log')
    path_logger = os.path.join(path_root, 'conf_mat', f'{os.path.basename(path_root)}_{experiment_name}.log')

    Logger.open(path_logger)

    metadata = pd.read_csv(path_metadata)

    # continue training from where we left off last run
    if os.path.isfile(path_report_conf_mat):
        df_report_conf_mat = pd.read_csv(path_report_conf_mat)
        n_report_conf_mat = len(df_report_conf_mat)
        last_model_name_in_report_conf_mat = df_report_conf_mat.iloc[-1]['model_name']
        print(f'Continue training (last id is {last_model_name_in_report_conf_mat})')
    else:
        print('Training from scratch')
        last_model_name_in_report_conf_mat = None
        n_report_conf_mat = 0
        df_report_conf_mat = pd.DataFrame(columns=[
            # preliminary info about the model
            'model_name', 'model_architecture', 'model_label', 'trigger_type_aux',

            # the features we compute
            'h_clean', 'kl_clean',
            'h_square30', 'kl_square30',
            'h_gotham', 'kl_gotham',
            'h_kelvin', 'kl_kelvin',
            'h_lomo', 'kl_lomo',
            'h_nashville', 'kl_nashville',
            'h_toaster', 'kl_toaster',

            # other data
            'trigger_color', 'num_classes'
        ])

    for _, row in metadata.iterrows():
        start_time = datetime.now()
        model_name = row['model_name']
        # model_id = int(model_name[3:])
        # if lim_left <= model_id <= lim_right:
        if (last_model_name_in_report_conf_mat is None) or (last_model_name_in_report_conf_mat is not None and model_name > last_model_name_in_report_conf_mat):
            model_label = 'backdoor' if row['poisoned'] else 'clean'
            model_architecture = row['model_architecture']

            trigger_type_aux = get_trigger_type_aux_value(row['trigger_type'], row['polygon_side_count'], row['instagram_filter_type'])
            num_classes = row['number_classes']

            ###############################################################################################################

            Logger.log()
            Logger.log(experiment_name)
            Logger.log(f'model {model_name} {model_architecture} ({model_label})')
            path_model = os.path.join(path_root, model_name)

            path_model_cnn = os.path.join(path_model, 'model.pt')
            model = torch.load(path_model_cnn, map_location=device).eval()

            # the keys will store the confusion distribution values for specific dataset
            # add it here in for-loop because I am deleting it at the end of the loop to save memory
            dict_dataset_h_kl = {
                'clean_example_data': None,
                square_dataset_name: None,
                'backdoored_data_filter_gotham': None,
                'backdoored_data_filter_kelvin': None,
                'backdoored_data_filter_lomo': None,
                'backdoored_data_filter_nashville': None,
                'backdoored_data_filter_toaster': None
            }

            for dataset_name in dict_dataset_h_kl:
                path_data = os.path.join(path_model, dataset_name)

                Logger.log(f'computing H and KL for {dataset_name}...', end='')
                dict_dataset_h_kl[dataset_name] = get_confusion_matrix_stats(model, device, path_data, batch_size)
                Logger.log('done')

            # compute mean and stds for confusion distributions
            h_clean, kl_clean = dict_dataset_h_kl['clean_example_data']
            h_square, kl_square = dict_dataset_h_kl[square_dataset_name]
            h_gotham, kl_gotham = dict_dataset_h_kl['backdoored_data_filter_gotham']
            h_kelvin, kl_kelvin = dict_dataset_h_kl['backdoored_data_filter_kelvin']
            h_lomo, kl_lomo = dict_dataset_h_kl['backdoored_data_filter_lomo']
            h_nashville, kl_nashville = dict_dataset_h_kl['backdoored_data_filter_nashville']
            h_toaster, kl_toaster = dict_dataset_h_kl['backdoored_data_filter_toaster']

            df_report_conf_mat.loc[n_report_conf_mat] = [
                # preliminary info about the model
                model_name, model_architecture, model_label, trigger_type_aux,

                h_clean, kl_clean,
                h_square, kl_square,
                h_gotham, kl_gotham,
                h_kelvin, kl_kelvin,
                h_lomo, kl_lomo,
                h_nashville, kl_nashville,
                h_toaster, kl_toaster,

                # other data
                trigger_color, num_classes
            ]
            n_report_conf_mat += 1
            df_report_conf_mat.to_csv(path_report_conf_mat, index=False)
            end_time = datetime.now()

            del model, dict_dataset_h_kl
            torch.cuda.empty_cache()

            Logger.log(f'model {model_name} took {end_time - start_time}')
    Logger.close()


if __name__ == '__main__':
    main()

# python3.7 -W ignore round_2.py
