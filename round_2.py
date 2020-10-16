import sys
import os
import ast
import math
import pickle
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

import tools.aux_funcs as af
import tools.model_funcs as mf
from architectures.SDNs.LightSDN import LightSDN

from tools.logistics import *
from tools.data import create_backdoored_dataset
from tools.logger import Logger


def get_trigger_type_aux_value(trigger_type, trigger_type_option):
    if trigger_type == 'instagram':
        return trigger_type_option.replace('XForm', '').replace('Filter', '')
    else:
        if trigger_type == 'None':
            return trigger_type
        else:
            return f'{trigger_type}-{trigger_type_option}'


def main():
    dict_arch_type = {
        'densenet': SDNConfig.DenseNet_blocks,
        'googlenet': SDNConfig.GoogLeNet,
        'inception': SDNConfig.Inception3,
        'mobilenet': SDNConfig.MobileNet2,
        'resnet': SDNConfig.ResNet,
        'shufflenet': SDNConfig.ShuffleNet,
        'squeezenet': SDNConfig.SqueezeNet,
        'vgg': SDNConfig.VGG,
        'wideresnet': SDNConfig.ResNet,
    }

    # parameters
    test_ratio = 0
    batch_size = 100  # for confusion experiment
    # device = 'cpu'
    device = af.get_pytorch_device()

    # experiment_name = 'square20-gotham-kelvin-lomo-nashville'
    experiment_name = 'square-5-10-15-20-25-30-35-40-45-50'

    # begin
    np.random.seed(666)
    path_root_project = get_project_root_path()
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round2-train-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')
    path_report = os.path.join(path_root, 'ics_svm', f'{os.path.basename(path_root)}_{experiment_name}.csv')
    path_logger = os.path.join(path_root, 'ics_svm', f'{os.path.basename(path_root)}_{experiment_name}.log')

    Logger.open(path_logger)

    metadata = pd.read_csv(path_metadata)

    # continue training from where we left off last run
    if os.path.isfile(path_report):
        df_report = pd.read_csv(path_report)
        n_report = len(df_report)
        last_model_name_in_report = df_report.iloc[-1]['model_name']
        print(f'Continue training (last id is {last_model_name_in_report})')
    else:
        print('Training from scratch')
        last_model_name_in_report = None
        n_report = 0
        df_report = pd.DataFrame(columns=[
            # preliminary info about the model
            'model_name', 'model_architecture', 'model_label', 'trigger_type_aux',

            # place differences here to visualize them easier
            'square5_mean_diff', 'square5_std_diff',
            'square10_mean_diff', 'square10_std_diff',
            'square15_mean_diff', 'square15_std_diff',
            'square20_mean_diff', 'square20_std_diff',
            'square25_mean_diff', 'square25_std_diff',
            'square30_mean_diff', 'square30_std_diff',
            'square35_mean_diff', 'square35_std_diff',
            'square40_mean_diff', 'square40_std_diff',
            'square45_mean_diff', 'square45_std_diff',
            'square50_mean_diff', 'square50_std_diff',
            # 'gotham_mean_diff', 'gotham_std_diff',
            # 'kelvin_mean_diff', 'kelvin_std_diff',
            # 'lomo_mean_diff', 'lomo_std_diff',
            # 'nashville_mean_diff', 'nashville_std_diff',
            # 'toaster_mean_diff', 'toaster_std_diff',

            # place effective metrics from confusion distribution
            'clean_mean', 'clean_std',
            'square5_mean', 'square5_std',
            'square10_mean', 'square10_std',
            'square15_mean', 'square15_std',
            'square20_mean', 'square20_std',
            'square25_mean', 'square25_std',
            'square30_mean', 'square30_std',
            'square35_mean', 'square35_std',
            'square40_mean', 'square40_std',
            'square45_mean', 'square45_std',
            'square50_mean', 'square50_std',
            # 'gotham_mean', 'gotham_std',
            # 'kelvin_mean', 'kelvin_std',
            # 'lomo_mean', 'lomo_std',
            # 'nashville_mean', 'nashville_std',
            # 'toaster_mean', 'toaster_std',

            # other data
            'trigger_color', 'num_classes'
        ])

    Logger.log('!!! Round 2: USE RGB COLORS')
    for _, row in metadata.iterrows():
        start_time = datetime.now()
        model_name = row['model_name']
        # model_id = int(model_name[3:])

        if last_model_name_in_report is not None and model_name > last_model_name_in_report:
            model_label = 'backdoor' if row['poisoned'] else 'clean'
            model_architecture = row['model_architecture']

            trigger_color = row['trigger_color']
            trigger_type = row['trigger_type']
            trigger_type_option = row['trigger_type_option']
            trigger_type_aux = get_trigger_type_aux_value(trigger_type, trigger_type_option)
            num_classes = row['number_classes']

            if trigger_color == 'None':
                trigger_color = (0, 0, 0)  # default color
            else:  # do not reverse the color
                trigger_color = tuple(ast.literal_eval(row['trigger_color'].replace(' ', ', ')))

            ###############################################################################################################

            Logger.log()
            Logger.log(experiment_name)
            Logger.log(f'model {model_name} {model_architecture} ({model_label})')
            path_model = os.path.join(path_root, model_name)

            Logger.log(f'loading model {model_name} ({model_label})...', end='')
            sdn_type = [v for k, v in dict_arch_type.items() if model_architecture.startswith(k)][0]
            path_model_cnn = os.path.join(path_model, 'model.pt')
            path_model_ics = os.path.join(path_model, 'svm', 'svm_models')
            sdn_light = LightSDN(path_model_cnn, path_model_ics, sdn_type, num_classes, device)
            Logger.log('done')

            # the keys will store the confusion distribution values for specific dataset
            # add it here in for because I am deleting it at the end of the loop to save memory
            dict_dataset_confusion = {
                'example_data': None,
                'backdoored_data_custom-square-size-5_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-10_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-15_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-20_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-25_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-30_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-35_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-40_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-45_backd-original-color_clean-black-color': None,
                'backdoored_data_custom-square-size-50_backd-original-color_clean-black-color': None
                # 'backdoored_data_filter_gotham': None,
                # 'backdoored_data_filter_kelvin': None,
                # 'backdoored_data_filter_lomo': None,
                # 'backdoored_data_filter_nashville': None,
                # 'backdoored_data_filter_toaster': None
            }

            # iterate through all backdoored datasets, compute and save the confusion scores
            for dataset_name in dict_dataset_confusion:
                path_data = os.path.join(path_model, dataset_name)

                Logger.log(f'reading dataset {dataset_name}...', end='')
                dataset = TrojAI(folder=path_data, test_ratio=test_ratio, batch_size=batch_size, device=device, opencv_format=False)
                Logger.log('done')

                Logger.log(f'computing confusion for {dataset_name}...', end='')
                dict_dataset_confusion[dataset_name] = mf.compute_confusion(sdn_light, dataset.train_loader, device)
                Logger.log('done')

            # compute mean and stds for confusion distributions
            clean_mean = np.mean(dict_dataset_confusion['example_data'])
            clean_std = np.std(dict_dataset_confusion['example_data'])

            square5_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-5_backd-original-color_clean-black-color'])
            square5_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-5_backd-original-color_clean-black-color'])

            square10_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-10_backd-original-color_clean-black-color'])
            square10_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-10_backd-original-color_clean-black-color'])

            square15_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-15_backd-original-color_clean-black-color'])
            square15_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-15_backd-original-color_clean-black-color'])

            square20_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-20_backd-original-color_clean-black-color'])
            square20_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-20_backd-original-color_clean-black-color'])

            square25_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-25_backd-original-color_clean-black-color'])
            square25_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-25_backd-original-color_clean-black-color'])

            square30_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-30_backd-original-color_clean-black-color'])
            square30_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-30_backd-original-color_clean-black-color'])

            square35_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-35_backd-original-color_clean-black-color'])
            square35_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-35_backd-original-color_clean-black-color'])

            square40_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-40_backd-original-color_clean-black-color'])
            square40_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-40_backd-original-color_clean-black-color'])

            square45_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-45_backd-original-color_clean-black-color'])
            square45_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-45_backd-original-color_clean-black-color'])

            square50_mean = np.mean(dict_dataset_confusion['backdoored_data_custom-square-size-50_backd-original-color_clean-black-color'])
            square50_std = np.std(dict_dataset_confusion['backdoored_data_custom-square-size-50_backd-original-color_clean-black-color'])

            # square_mean = np.mean(dict_dataset_confusion[square_dataset_name])
            # square_std = np.std(dict_dataset_confusion[square_dataset_name])

            # gotham_mean = np.mean(dict_dataset_confusion['backdoored_data_filter_gotham'])
            # gotham_std = np.std(dict_dataset_confusion['backdoored_data_filter_gotham'])
            #
            # kelvin_mean = np.mean(dict_dataset_confusion['backdoored_data_filter_kelvin'])
            # kelvin_std = np.std(dict_dataset_confusion['backdoored_data_filter_kelvin'])
            #
            # lomo_mean = np.mean(dict_dataset_confusion['backdoored_data_filter_lomo'])
            # lomo_std = np.std(dict_dataset_confusion['backdoored_data_filter_lomo'])
            #
            # nashville_mean = np.mean(dict_dataset_confusion['backdoored_data_filter_nashville'])
            # nashville_std = np.std(dict_dataset_confusion['backdoored_data_filter_nashville'])

            # toaster_mean = np.mean(dict_dataset_confusion['backdoored_data_filter_toaster'])
            # toaster_std = np.std(dict_dataset_confusion['backdoored_data_filter_toaster'])

            # # compute differences for mean and stds between backdoored and clean
            square5_mean_diff = square5_mean - clean_mean
            square5_std_diff = square5_std - clean_std

            square10_mean_diff = square10_mean - clean_mean
            square10_std_diff = square10_std - clean_std

            square15_mean_diff = square15_mean - clean_mean
            square15_std_diff = square15_std - clean_std

            square20_mean_diff = square20_mean - clean_mean
            square20_std_diff = square20_std - clean_std

            square25_mean_diff = square25_mean - clean_mean
            square25_std_diff = square25_std - clean_std

            square30_mean_diff = square30_mean - clean_mean
            square30_std_diff = square30_std - clean_std

            square35_mean_diff = square35_mean - clean_mean
            square35_std_diff = square35_std - clean_std

            square40_mean_diff = square40_mean - clean_mean
            square40_std_diff = square40_std - clean_std

            square45_mean_diff = square45_mean - clean_mean
            square45_std_diff = square45_std - clean_std

            square50_mean_diff = square50_mean - clean_mean
            square50_std_diff = square50_std - clean_std

            # square_mean_diff = square_mean - clean_mean
            # square_std_diff = square_std - clean_std
            #
            # gotham_mean_diff = gotham_mean - clean_mean
            # gotham_std_diff = gotham_std - clean_std
            #
            # kelvin_mean_diff = kelvin_mean - clean_mean
            # kelvin_std_diff = kelvin_std - clean_std
            #
            # lomo_mean_diff = lomo_mean - clean_mean
            # lomo_std_diff = lomo_std - clean_std
            #
            # nashville_mean_diff = nashville_mean - clean_mean
            # nashville_std_diff = nashville_std - clean_std

            # toaster_mean_diff = toaster_mean - clean_mean
            # toaster_std_diff = toaster_std - clean_std

            df_report.loc[n_report] = [
                # preliminary info about the model
                model_name, model_architecture, model_label, trigger_type_aux,

                ## place differences here to visualize them easier
                # square_mean_diff, square_std_diff,
                square5_mean_diff, square5_std_diff,
                square10_mean_diff, square10_std_diff,
                square15_mean_diff, square15_std_diff,
                square20_mean_diff, square20_std_diff,
                square25_mean_diff, square25_std_diff,
                square30_mean_diff, square30_std_diff,
                square35_mean_diff, square35_std_diff,
                square40_mean_diff, square40_std_diff,
                square45_mean_diff, square45_std_diff,
                square50_mean_diff, square50_std_diff,
                # gotham_mean_diff, gotham_std_diff,
                # kelvin_mean_diff, kelvin_std_diff,
                # lomo_mean_diff, lomo_std_diff,
                # nashville_mean_diff, nashville_std_diff,
                # toaster_mean_diff, toaster_std_diff,

                ## place effective metrics from confusion distribution
                clean_mean, clean_std,
                square5_mean, square5_std,
                square10_mean, square10_std,
                square15_mean, square15_std,
                square20_mean, square20_std,
                square25_mean, square25_std,
                square30_mean, square30_std,
                square35_mean, square35_std,
                square40_mean, square40_std,
                square45_mean, square45_std,
                square50_mean, square50_std,
                # square_mean, square_std,
                # gotham_mean, gotham_std,
                # kelvin_mean, kelvin_std,
                # lomo_mean, lomo_std,
                # nashville_mean, nashville_std,
                # toaster_mean, toaster_std,

                # other data
                trigger_color, num_classes
            ]
            n_report += 1
            df_report.to_csv(path_report, index=False)
            end_time = datetime.now()

            del sdn_light, dict_dataset_confusion
            torch.cuda.empty_cache()

            Logger.log(f'model {model_name} took {end_time - start_time}')
    Logger.close()


if __name__ == '__main__':
    main()

# python3.7 -W ignore round_2.py
