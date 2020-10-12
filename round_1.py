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
from tools.network_architectures import load_trojai_model
from architectures.SDNs.SDNConfig import SDNConfig

from tools.logistics import *
from tools.data import create_backdoored_dataset
from tools.logger import Logger


def main():
    trigger_size = sys.argv[1] # save it in this variable to be used in exp_desc. Later, overwrite it
    modified_trigger_size = int(trigger_size)
    device = sys.argv[2]

    dict_arch_type = {
        'densenet': SDNConfig.DenseNet_blocks,
        'resnet': SDNConfig.ResNet,
        'inception': SDNConfig.Inception3
    }

    # parameters
    test_ratio = 0
    batch_size = 100 # for confusion experiment
    # device = 'cpu'
    # device = af.get_pytorch_device()
    
    # sdn_name = 'ics_train100_test0_bs25'
    # cnn_name = 'model.pt'
    path_trigger = 'square'

    exp_desc = f'custom-square-size-{trigger_size}_backd-original-color_clean-black-color'
    # exp_desc = '{working_architecture}_sqrt-size_backd-original-color_clean-black-color'
    # exp_desc = '{working_architecture}_sqrt-size_backd-black-color_clean-black-color'
    # exp_desc = '{working_architecture}_original-size_backd-original-color_clean-black-color'
    # exp_desc = '{working_architecture}_original-size_backd-black-color_clean-black-color'
    # exp_desc = f'{working_architecture}_custom-square-size-{trigger_size}_backd-black-color_clean-black-color'
    # exp_desc = f'{working_architecture}_custom-square-size-{trigger_size}_backd-original-color_clean-black-color'

    # begin
    np.random.seed(666)
    path_root_project = get_project_root_path()
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-dataset-train')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-holdout-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')
    path_report = os.path.join(path_root, 'ics_svm', f'{os.path.basename(path_root)}_{exp_desc}.csv')

    path_logger = os.path.join(path_root, 'ics_svm', f'{os.path.basename(path_root)}_{exp_desc}.log')
    Logger.open(path_logger)

    if 'train' in os.path.basename(path_root): # append 'models' for training dataset
        path_root = os.path.join(path_root, 'models')

    metadata = pd.read_csv(path_metadata)

    if os.path.isfile(path_report):
        df_report = pd.read_csv(path_report)
        n_report = len(df_report)
        last_model_name_in_report = df_report.iloc[-1]['model_name']
    else:
        last_model_name_in_report = None
        n_report = 0
        df_report = pd.DataFrame(columns=[
            'model_name',
            'model_architecture',
            'model_label',
            'trigger_color',
            'original_trigger_size',
            'our_trigger_size',
            'num_classes',
            'clean_mean',
            'clean_std',
            'backd_mean',
            'backd_std'
        ])

    Logger.log('!!! Round 1: reversing trigger color')
    for _, row in metadata.iterrows():
        start_time = datetime.now()

        model_name = row['model_name']
        if last_model_name_in_report is not None and model_name > last_model_name_in_report:
            model_architecture = row['model_architecture']
            model_name_int = int(model_name[3:])
            num_classes = row['number_classes']
            ground_truth = row['ground_truth']
            model_label = 'backdoor' if ground_truth else 'clean'

            trigger_color = row['trigger_color']
            triggered_classes = ast.literal_eval(row['triggered_classes'].replace(' ', ', '))
            trigger_target_class = row['trigger_target_class']
            trigger_target_class = int(trigger_target_class) if trigger_target_class != 'None' else 0 # default class
            trigger_size = row['trigger_size']

            if len(triggered_classes) == 0:
                triggered_classes = list(range(num_classes))

            if trigger_color == 'None':
                trigger_color = (0, 0, 0) # default color
            else: # reversed because round 1 uses BGR
                trigger_color = tuple(reversed(ast.literal_eval(row['trigger_color'].replace(' ', ', '))))

            ###############################################################################################################

            Logger.log()
            Logger.log(exp_desc)
            Logger.log(f'model {model_name} {model_architecture} ({model_label})')
            path_model = os.path.join(path_root, model_name)

            path_data_clean = os.path.join(path_model, 'example_data')
            path_data_backd = os.path.join(path_model, f'backdoored_data_{exp_desc}')

            if os.path.isdir(path_data_backd) and len(os.listdir(path_data_backd)) > 0:
                # shutil.rmtree(path_data_backd)
                Logger.log(f'backdoored data folder already exists {path_data_backd}')
            else:
                Logger.log(f'creating backdoored dataset for {model_name}...', end='')
                create_backdoored_dataset(dir_clean_data=path_data_clean,
                                          dir_backdoored_data=path_data_backd,
                                          filename_trigger=path_trigger,
                                          triggered_fraction=1.0, # only polygon trigger
                                          triggered_classes=triggered_classes,
                                          trigger_target_class=trigger_target_class,
                                          trigger_color=trigger_color,
                                          trigger_size=modified_trigger_size, # comes from commandline
                                          p_trigger=1.0,
                                          keep_original=False)
                Logger.log('done')

            Logger.log('reading clean & backdoored datasets...', end='')
            dataset_clean = TrojAI(folder=path_data_clean, test_ratio=test_ratio, batch_size=batch_size, device=device)
            dataset_backd = TrojAI(folder=path_data_backd, test_ratio=test_ratio, batch_size=batch_size, device=device)
            Logger.log('done')

            Logger.log(f'loading model {model_name} ({model_label})...', end='')
            # sdn_model = load_trojai_model(path_model, sdn_name, cnn_name, num_classes, sdn_type, device)
            # sdn_model = sdn_model.eval().to(device)
            sdn_type = [v for k, v in dict_arch_type.items() if model_architecture.startswith(k)][0]

            path_model_cnn = os.path.join(path_model, 'model.pt')
            path_model_ics = os.path.join(path_model, 'svm', 'svm_models')

            sdn_light = LightSDN(path_model_cnn, path_model_ics, sdn_type, num_classes, device)
            Logger.log('done')

            # IMPORTANT: when test_ratio = 0, train_loader = test_loader
            Logger.log(f'computing confusion for clean data...', end='')
            confusion_clean = mf.compute_confusion(sdn_light, dataset_clean.train_loader, device)
            Logger.log('done')

            Logger.log(f'computing confusion for backdoored data...', end='')
            confusion_backd = mf.compute_confusion(sdn_light, dataset_backd.train_loader, device)
            Logger.log('done')

            path_confusion = os.path.join(path_model, 'confusion')
            af.create_path(path_confusion)
            af.save_obj(confusion_clean, os.path.join(path_confusion, f'{model_name}_conf_clean_{exp_desc}'))
            af.save_obj(confusion_backd, os.path.join(path_confusion, f'{model_name}_conf_backd_{exp_desc}'))

            clean_mean = np.mean(confusion_clean)
            clean_std = np.std(confusion_clean)
            backd_mean = np.mean(confusion_backd)
            backd_std = np.std(confusion_backd)

            df_report.loc[n_report] = [
                model_name,
                model_architecture,
                model_label,
                trigger_color,
                trigger_size,
                modified_trigger_size,
                num_classes,
                clean_mean,
                clean_std,
                backd_mean,
                backd_std
            ]
            n_report += 1
            df_report.to_csv(path_report, index=False)
            end_time = datetime.now()

            del sdn_light, dataset_clean, dataset_backd, confusion_clean, confusion_backd
            torch.cuda.empty_cache()

            Logger.log(f'model {model_name} took {end_time - start_time}')
    Logger.close()


if __name__ == '__main__':
    main()

# python3.7 -W ignore round_1.py 5 cuda:0