import ast
import sys
import pickle
import numpy as np
import pandas as pd

import tools.aux_funcs as af
import tools.model_funcs as mf
from tools.network_architectures import load_trojai_model
from architectures.SDNs.SDNConfig import SDNConfig

from tools.logistics import *
from tools.data import create_backdoored_dataset


def find_most_frequently_used_trigger(path_root, metadata):
    dict_freq = {}
    for _, row in metadata.iterrows():
        if row['ground_truth']:
            model_name = row['model_name']
            lookup_dir = os.path.join(path_root, model_name, 'triggers')
            if os.path.isdir(lookup_dir):
                triggers = os.listdir(lookup_dir)
                for trig in triggers:
                    if trig not in dict_freq:
                        dict_freq[trig] = [model_name]
                    else:
                        dict_freq[trig].append(model_name)
    best_size = 0
    best_trig = None
    for trig in dict_freq:
        print(f'{trig} => {dict_freq[trig]}')
        size = len(dict_freq[trig])
        if size > best_size:
            best_size = size
            best_trig = trig
    best_trigger_path = os.path.join(path_root, dict_freq[best_trig][0], 'triggers', best_trig)
    return best_trigger_path


def main():
    # parameters
    torch.cuda.empty_cache()
    test_ratio = 0
    batch_size = 128 # for confusion experiment
    device = 'cpu'
    # device = af.get_pytorch_device()
    sdn_name = 'ics_train100_test0_bs25'
    cnn_name = 'model.pt'
    sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks

    # begin
    np.random.seed(666)
    path_root_project = get_project_root_path()
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-dataset-train')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round1-holdout-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')
    path_report = os.path.join(path_root, f'{os.path.basename(path_root)}-report-black-square.csv')

    if 'train' in os.path.basename(path_root): # append 'models' for training dataset
        path_root = os.path.join(path_root, 'models')

    metadata = pd.read_csv(path_metadata)

    # most_frequent_trigger_path = find_most_frequently_used_trigger(path_root, metadata)

    available_architectures = ['densenet121']

    df_report = pd.DataFrame(columns=['model_name', 'model_label', 'clean_mean', 'clean_std', 'backd_mean', 'backd_std'])
    n_report = 0

    n_models_clean = 0
    n_models_backd = 0
    print('!!! Round 1: reversing trigger color')
    for _, row in metadata.iterrows():
        model_architecture = row['model_architecture']
        model_name = row['model_name']
        num_classes = row['number_classes']
        ground_truth = row['ground_truth']
        model_label = 'backdoor' if ground_truth else 'clean'
        trigger_color = row['trigger_color']
        if trigger_color == 'None':
            trigger_color = (0, 0, 0) # default color
        else: # reversed because round 1 uses BGR
            trigger_color = tuple(reversed(ast.literal_eval(row['trigger_color'].replace(' ', ', '))))
        triggered_classes = ast.literal_eval(row['triggered_classes'].replace(' ', ', '))
        if len(triggered_classes) == 0:
            triggered_classes = list(range(num_classes))
        trigger_target_class = row['trigger_target_class']
        trigger_target_class = int(trigger_target_class) if trigger_target_class != 'None' else 0 # default class for clean models
        trigger_size = row['trigger_size']
        trigger_size = int(trigger_size) if trigger_size != 'None' else 30 # default value

        if model_architecture in available_architectures:
            print()
            print(f'model {model_name} ({model_label})')
            path_model = os.path.join(path_root, model_name)

            path_data_clean = os.path.join(path_model, 'example_data')
            path_data_backd = os.path.join(path_model, 'example_data_backdoored')

            if os.path.isdir(path_data_backd):# or len(os.listdir(path_data_backd)) < 2:
                os.remove(path_data_backd)
            # if ground_truth: # model is backdoored
            #     dir_triggers = os.path.join(path_model, 'triggers')
            #     trigger_name = os.listdir(dir_triggers)[0] # assume only one trigger (check this for round > 1)
            #     path_trigger = os.path.join(dir_triggers, trigger_name)
            # else: # model is clean
            #     path_trigger = most_frequent_trigger_path

            path_trigger = 'square'
            trigger_color = (0, 0, 0)

            print(f'creating backdoored dataset for {model_name}...', end=''); sys.stdout.flush()
            create_backdoored_dataset(dir_clean_data=path_data_clean,
                                      dir_backdoored_data=path_data_backd,
                                      filename_trigger=path_trigger,
                                      triggered_fraction=1.0, # only polygon trigger
                                      triggered_classes=triggered_classes,
                                      trigger_target_class=trigger_target_class,
                                      trigger_color=trigger_color,
                                      trigger_size=trigger_size,
                                      p_trigger=1.0,
                                      keep_original=False)
            print('done')

            print('reading clean & backdoored datasets...', end=''); sys.stdout.flush()
            dataset_clean = TrojAI(folder=path_data_clean, test_ratio=test_ratio, batch_size=batch_size, device=device)
            dataset_backd = TrojAI(folder=path_data_backd, test_ratio=test_ratio, batch_size=batch_size, device=device)
            print('done')

            print(f'loading model {model_name} ({model_label})...', end=''); sys.stdout.flush()
            sdn_model = load_trojai_model(path_model, sdn_name, cnn_name, num_classes, sdn_type, device)
            sdn_model = sdn_model.eval()
            print('done')

            # IMPORTANT: when test_ratio = 0, train_loader = test_loader
            print(f'computing confusion for clean data...', end=''); sys.stdout.flush()
            confusion_clean = mf.compute_confusion(sdn_model, dataset_clean.train_loader, device)
            print('done')

            print(f'computing confusion for backdoored data...', end=''); sys.stdout.flush()
            confusion_backd = mf.compute_confusion(sdn_model, dataset_backd.train_loader, device)
            print('done')

            path_confusion = os.path.join(path_model, 'confusion')
            af.create_path(path_confusion)
            af.save_obj(confusion_clean, os.path.join(path_confusion, f'{model_name}_confusion_clean_data'))
            af.save_obj(confusion_backd, os.path.join(path_confusion, f'{model_name}_confusion_backdoored_data'))

            clean_mean = np.mean(confusion_clean)
            clean_std = np.std(confusion_clean)
            backd_mean = np.mean(confusion_backd)
            backd_std = np.std(confusion_backd)

            # ['model_name', 'model_label', 'clean_mean', 'clean_std', 'backd_mean', 'backd_std']
            df_report.loc[n_report] = [model_name, model_label, clean_mean, clean_std, backd_mean, backd_std]
            n_report += 1
            df_report.to_csv(path_report, index=False)

            n_models_clean += int(ground_truth == False)
            n_models_backd += int(ground_truth == True)
            if n_models_clean == 5 and n_models_backd == 5:
                break



if __name__ == '__main__':
    main()
    print('script ended')
