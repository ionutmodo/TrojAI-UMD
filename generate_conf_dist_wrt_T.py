# import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
# import torch.nn.functional as F

# import tools.aux_funcs as af
from tools.logger import Logger
from tools.logistics import *
# from round_features_sdn import get_trigger_type_aux_value
# from notebooks.methods import encode_architecture

"""
Available keys for npz files when num_ics = 4:
    'num_ics',
    'logit_ic_0',
    'logit_ic_1',
    'logit_ic_2',
    'logit_ic_3', # if num_ics=3, then this key won't exist
    'logit_net_out',
    'conf_mat_ic0',
    'conf_mat_ic1',
    'conf_mat_ic2',
    'conf_mat_ic3', # if num_ics=3, then this key won't exist
    'conf_dist_original'
"""


# this method can also be found in notebooks/methods
def encode_architecture(model_architecture):
    arch_codes = ['densenet', 'googlenet', 'inception', 'mobilenet', 'resnet', 'shufflenet', 'squeezenet', 'vgg']
    for index, arch in enumerate(arch_codes):
        if arch in model_architecture:
            return index
    return None


# this method can also be found in notebooks/methods
def encode_backdoor(trigger_type_aux):
    code = None
    if trigger_type_aux == 'none':
        code = 0
    elif 'polygon' in trigger_type_aux:
        code = 1
    elif 'gotham' in trigger_type_aux:
        code = 2
    elif 'kelvin' in trigger_type_aux:
        code = 3
    elif 'lomo' in trigger_type_aux:
        code = 4
    elif 'nashville' in trigger_type_aux:
        code = 5
    elif 'toaster' in trigger_type_aux:
        code = 6
    return code


# this method can also be found in round_features_sdn
def get_trigger_type_aux_value(triggers_0_type, triggers_0_instagram_filter_type, triggers_1_type, triggers_1_instagram_filter_type):
    triggers_0_type = triggers_0_type.lower()
    triggers_1_type = triggers_1_type.lower()
    triggers_0_instagram_filter_type = triggers_0_instagram_filter_type.lower().replace('filter', '').replace('xform', '')
    triggers_1_instagram_filter_type = triggers_1_instagram_filter_type.lower().replace('filter', '').replace('xform', '')

    if triggers_0_type == 'instagram':
        backd_0_str = f'instagram-{triggers_0_instagram_filter_type}'
        backd_0_code = encode_backdoor(triggers_0_instagram_filter_type)
    else:
        backd_0_str = triggers_0_type
        backd_0_code = encode_backdoor(triggers_0_type)

    if triggers_1_type == 'instagram':
        backd_1_str = f'instagram-{triggers_1_instagram_filter_type}'
        backd_1_code = encode_backdoor(triggers_1_instagram_filter_type)
    else:
        backd_1_str = triggers_1_type
        backd_1_code = encode_backdoor(triggers_1_type)

    return f'{backd_0_str}_{backd_1_str}', backd_0_code, backd_1_code


def main():
    T = 0.0
    device = af.get_pytorch_device()

    list_limits = {
        'windows10': (0, 1007),
        'opensub03.umiacs.umd.edu': (0, 1007),
        'openlab08.umiacs.umd.edu': (0, 1007),
        # 'openlab30.umiacs.umd.edu': (0, 249),
        # 'openlab31.umiacs.umd.edu': (250, 499),
        # 'openlab32.umiacs.umd.edu': (500, 749),
        # 'openlab33.umiacs.umd.edu': (750, 1007),
    }
    if len(sys.argv) != 3:
        lim_left, lim_right = list_limits[socket.gethostname()]
    else:
        lim_left, lim_right = int(sys.argv[1]), int(sys.argv[2])

    print(f'lim_left={lim_left}, lim_right={lim_right}')
    experiment_name = f'fc_synthetic_polygon-all-gray_filters_T={T:.1f}_{lim_left}-{lim_right}'

    path_root_project = get_project_root_path()
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round4-train-dataset')

    path_report_conf_dist = os.path.join(path_root, 'ics_fc', f'{os.path.basename(path_root)}_{experiment_name}.csv')

    synthetic_stats_root = os.path.join(path_root, 'synthetic_stats')

    metadata_root = os.path.join(path_root, 'METADATA.csv')
    metadata = pd.read_csv(metadata_root)

    path_logger = os.path.join(path_root, 'ics_fc', f'{os.path.basename(path_root)}_{experiment_name}.log')

    Logger.open(path_logger)

    # continue training from where we left off last run
    if os.path.isfile(path_report_conf_dist):
        df_report_conf_dist = pd.read_csv(path_report_conf_dist)
        n_report_conf_dist = len(df_report_conf_dist)
        last_model_name_in_report_conf_dist = df_report_conf_dist.iloc[-1]['model_name']
        print(f'Continue training (last id is {last_model_name_in_report_conf_dist})')
    else:
        print('Training from scratch')
        last_model_name_in_report_conf_dist = None
        n_report_conf_dist = 0
        df_report_conf_dist = pd.DataFrame(columns=[
            # preliminary info about the model
            'model_name', 'model_architecture', 'architecture_code', 'backdoor_code_0', 'backdoor_code_1', 'model_label', 'backdoor_string',

            'polygon_mean_diff', 'polygon_std_diff',
            'gotham_mean_diff', 'gotham_std_diff',
            'kelvin_mean_diff', 'kelvin_std_diff',
            'lomo_mean_diff', 'lomo_std_diff',
            'nashville_mean_diff', 'nashville_std_diff',
            'toaster_mean_diff', 'toaster_std_diff',

            'clean_mean', 'clean_std',
            'polygon_mean', 'polygon_std',
            'gotham_mean', 'gotham_std',
            'kelvin_mean', 'kelvin_std',
            'lomo_mean', 'lomo_std',
            'nashville_mean', 'nashville_std',
            'toaster_mean', 'toaster_std',

            # other data
            'num_classes',
            'counts'
        ])

    for index, row in metadata.iterrows():
        start_time = datetime.now()
        model_name = row['model_name']
        model_id = int(model_name[3:])

        if lim_left <= model_id <= lim_right and ((last_model_name_in_report_conf_dist is None) or (last_model_name_in_report_conf_dist is not None and model_name > last_model_name_in_report_conf_dist)):
            # wait for the round_features_sdn process to create the npz files for each model
            lookup_folder = os.path.join(synthetic_stats_root, model_name)
            while True:
                if not os.path.isdir(lookup_folder):
                    Logger.log(f'Waiting for folder synthetic_stats/{model_name} to be created...')
                    time.sleep(60)
                    continue
                files_count = len(os.listdir(lookup_folder))
                if files_count == 7:
                    time.sleep(10)
                    break
                Logger.log(f'Waiting for folder synthetic_stats/{model_name} to have 7 files...')
                time.sleep(60)

            backd_str, backd_0_code, backd_1_code = get_trigger_type_aux_value(row['triggers_0_type'], row['triggers_0_instagram_filter_type'], row['triggers_1_type'], row['triggers_1_instagram_filter_type'])
            model_label = 'backdoor' if row['poisoned'] else 'clean'
            model_architecture = row['model_architecture']
            num_classes = row['number_classes']
            architecture_code = encode_architecture(model_architecture)

            counts = {}
            conf_dist_all = {}
            conf_dist_T = {}

            for trigger_name in ['clean', 'polygon_all', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
                stats = np.load(os.path.join(synthetic_stats_root, model_name, f'{model_name}_{trigger_name}.npz'))
                num_ics = stats['num_ics'][0]

                confusion_all, confusion_T = [], []
                counts[trigger_name] = {ic: 0 for ic in range(num_ics+1)}
                for index_image in range(1000): # we have 1000 synthetic images
                    score_all, score_T, count_ics_T = 0.0, 0.0, 0

                    logit_net_out = torch.tensor(stats['logit_net_out'][index_image], device=device).unsqueeze(0)
                    softmax_net_out = F.softmax(logit_net_out)
                    for index_ic in range(num_ics):
                        logit_ic = torch.tensor(stats[f'logit_ic_{index_ic}'][index_image], device=device).unsqueeze(0)
                        softmax_ic = F.softmax(logit_ic)

                        dist = F.pairwise_distance(softmax_ic, softmax_net_out, p=1).item()
                        max_confidence = softmax_ic.max(1)[0].item()

                        score_all += dist
                        if max_confidence > T:
                            score_T += dist
                            count_ics_T += 1
                    counts[trigger_name][count_ics_T] += 1
                    confusion_all.append(score_all)
                    confusion_T.append(score_T)
                    # print(f'{trigger_name}\t#{index_image}\tall:{score_all:.3f}\tT:{score_T:.3f}\tcount:{count_ics_T} of {num_ics}')
                conf_dist_all[trigger_name] = np.array(confusion_all)
                conf_dist_T[trigger_name] = np.array(confusion_T)

            # compute mean and stds for confusion distributions
            clean_mean = np.mean(conf_dist_T['clean'])
            clean_std = np.std(conf_dist_T['clean'])

            polygon_mean = np.mean(conf_dist_T['polygon_all'])
            polygon_std = np.std(conf_dist_T['polygon_all'])

            gotham_mean = np.mean(conf_dist_T['gotham'])
            gotham_std = np.std(conf_dist_T['gotham'])

            kelvin_mean = np.mean(conf_dist_T['kelvin'])
            kelvin_std = np.std(conf_dist_T['kelvin'])

            lomo_mean = np.mean(conf_dist_T['lomo'])
            lomo_std = np.std(conf_dist_T['lomo'])

            nashville_mean = np.mean(conf_dist_T['nashville'])
            nashville_std = np.std(conf_dist_T['nashville'])

            toaster_mean = np.mean(conf_dist_T['toaster'])
            toaster_std = np.std(conf_dist_T['toaster'])

            ############ compute differences for mean and stds between backdoored and clean

            polygon_mean_diff = polygon_mean - clean_mean
            polygon_std_diff = polygon_std - clean_std

            gotham_mean_diff = gotham_mean - clean_mean
            gotham_std_diff = gotham_std - clean_std

            kelvin_mean_diff = kelvin_mean - clean_mean
            kelvin_std_diff = kelvin_std - clean_std

            lomo_mean_diff = lomo_mean - clean_mean
            lomo_std_diff = lomo_std - clean_std

            nashville_mean_diff = nashville_mean - clean_mean
            nashville_std_diff = nashville_std - clean_std

            toaster_mean_diff = toaster_mean - clean_mean
            toaster_std_diff = toaster_std - clean_std

            df_report_conf_dist.loc[n_report_conf_dist] = [
                # preliminary info about the model
                model_name, model_architecture, architecture_code, backd_0_code, backd_1_code, model_label, backd_str,

                polygon_mean_diff, polygon_std_diff,
                gotham_mean_diff, gotham_std_diff,
                kelvin_mean_diff, kelvin_std_diff,
                lomo_mean_diff, lomo_std_diff,
                nashville_mean_diff, nashville_std_diff,
                toaster_mean_diff, toaster_std_diff,

                ## place effective metrics from confusion distribution
                clean_mean, clean_std,
                polygon_mean, polygon_std,
                gotham_mean, gotham_std,
                kelvin_mean, kelvin_std,
                lomo_mean, lomo_std,
                nashville_mean, nashville_std,
                toaster_mean, toaster_std,

                # other data
                num_classes,
                str(counts)
            ]
            n_report_conf_dist += 1
            df_report_conf_dist.to_csv(path_report_conf_dist, index=False)
            end_time = datetime.now()

            del conf_dist_T
            torch.cuda.empty_cache()

            Logger.log(f'model {model_name} took {end_time - start_time}')

            # for trigger_name in ['clean', 'polygon_all', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
            #     mean_all, std_all = np.mean(conf_dist_all[trigger_name]), np.std(conf_dist_all[trigger_name])
            #     mean_T, std_T = np.mean(conf_dist_T[trigger_name]), np.std(conf_dist_T[trigger_name])
            #     print(f'{trigger_name} all={mean_all:.4f},{std_all:.4f}, T={mean_T:.4f},{std_T:.4f}')
            # print(counts)


if __name__ == '__main__':
    main()
