import os
import numpy as np
import pandas as pd
import torch

"""
Available keys for npz files when num_ics = 4:
    'num_ics',
    'logit_ic_0',
    'logit_ic_1',
    'logit_ic_2',
    'logit_ic_3',
    'logit_net_out',
    'conf_mat_ic0',
    'conf_mat_ic1',
    'conf_mat_ic2',
    'conf_mat_ic3',
    'conf_dist_original'
"""


def metric(softmax_clean, softmax_triggered):
    return 0 # fill in your desired metric (CrossEntropy)


synthetic_stats_root = '/fs/sdsatumd/ionmodo/TrojAI/TrojAI-data/round4-train-dataset/synthetic_stats'
metadata_root = '/fs/sdsatumd/ionmodo/TrojAI/TrojAI-data/round4-train-dataset/METADATA.csv'

metadata_csv = pd.read_csv(metadata_root)
metadata_csv = metadata_csv.set_index('model_name') # so we can do metadata_csv['id-00000000']

for model_id in range(1008): # loop through all models
    model_name = f'id-{model_id:08d}'
    model_label = 1 if bool(metadata_csv['poisoned'].loc[model_name]) == True else 0 # clean=0, backdoored=1

    stats_clean = np.load(os.path.join(synthetic_stats_root, model_name, f'{model_name}_clean.npz')) # load the clean stats
    stats_triggered = {} # load all triggered stats here
    for trigger_name in ['polygon_all', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
        stats_triggered[trigger_name] = np.load(os.path.join(synthetic_stats_root, model_name, f'{model_name}_{trigger_name}.npz'))

    num_ic = stats_clean['num_ics']
    for index_image in range(1000): # we have 1000 synthetic images
        for index_ic in range(num_ic): # loop through ICs
            # compute clean softmax for IC[index_ic] on image index_image
            logit_clean = stats_clean[f'logit_ic_{index_ic}'][index_image]
            softmax_clean = torch.nn.functional.softmax(logit_clean, dim=1)
            for trigger_name in ['polygon_all', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
                # compute trigger softmax for IC[index_ic] on image index_image
                logit_triggered = stats_triggered[f'logit_ic_{index_ic}'][index_image]
                softmax_triggered = torch.nn.functional.softmax(logit_triggered, dim=1)

                diff = metric(softmax_clean, softmax_triggered)

                # do whatever you want with this diff...
