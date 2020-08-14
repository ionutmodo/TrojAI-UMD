import os
import socket

import numpy as np
import torch

import tools.aux_funcs as af
import tools.model_funcs as mf
import tools.network_architectures as arcs
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNDenseNet121 import SDNDenseNet121
from settings import get_project_root_path
from tools.data import TrojAI


def proposal_plots(model_path, model_id, clean_dataset_dir, backdoored_dataset_dir, suffix, message, test_ratio, sdn_type, device='cpu'):
    sdn_name = f'ics_{suffix}'

    sdn_model, sdn_params = arcs.load_model(model_path, sdn_name, epoch=-1)
    sdn_model = sdn_model.to(device)

    plots_dir = f'confusion_experiments/proposal_plots'
    af.create_path(plots_dir)

    cnn_model = torch.load(os.path.join(model_path, 'model.pt')).to(device)
    cnn_model = SDNDenseNet121(cnn_model, (1, 3, 224, 224), 5, sdn_type, device)
    sdn_model.set_model(cnn_model)

    batch_size = 128
    clean_dataset = TrojAI(folder=clean_dataset_dir, test_ratio=test_ratio, batch_size=batch_size, device=device)
    backdoored_dataset = TrojAI(folder=backdoored_dataset_dir, test_ratio=test_ratio, batch_size=batch_size, device=device)

    print('CNN test on clean dataset:', mf.cnn_test(cnn_model, clean_dataset.test_loader, device))
    print('CNN test on backdoored dataset:', mf.cnn_test(cnn_model, backdoored_dataset.test_loader, device))
    print('SDN test on clean dataset:', mf.sdn_test(sdn_model, clean_dataset.test_loader, device))
    print('SDN test on backdoored dataset:', mf.sdn_test(sdn_model, backdoored_dataset.test_loader, device))

    clean_mean, clean_std = mf.sdn_confusion_stats(sdn_model, loader=clean_dataset.train_loader, device=device)
    print(f'clean confusion: mean={clean_mean}, std={clean_std}')

    clean_confusion_scores = mf.compute_confusion(sdn_model, clean_dataset.test_loader, device)
    clean_confusion_scores = (clean_confusion_scores - clean_mean) / clean_std

    backdoored_confusion_scores = mf.compute_confusion(sdn_model, backdoored_dataset.test_loader, device)
    backdoored_mean, backdoored_std = backdoored_confusion_scores.mean(), backdoored_confusion_scores.std()
    print(f'backdoored confusion: mean={backdoored_mean}, std={backdoored_std}')
    backdoored_confusion_scores = (backdoored_confusion_scores - clean_mean) / clean_std # divide backdoored by clean mean/std!

    af.overlay_two_histograms(save_path=plots_dir,
                              save_name=f'proposal_{model_id}_{sdn_name}',
                              hist_first_values=clean_confusion_scores,
                              hist_second_values=backdoored_confusion_scores,
                              first_label='w/o Trigger',
                              second_label='w Trigger',
                              xlabel='Confusion score',
                              title=message)

def main():
    # device = af.get_pytorch_device()
    device = 'cpu'

    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')

    suffix = 'train100_test0_bs25'
    test_ratio = 0

    for _id, _description in [(9, 'backdoored')]:
        model_id = f'id-{_id:08d}'
        print(f'----------{model_id} ({_description})----------')
        model_path = os.path.join(root_path, model_id)
        clean_dataset_dir = os.path.join(model_path, 'example_data')
        backdoored_dataset_dir = os.path.join(model_path, 'example_data_backdoored')
        proposal_plots(model_path,
                       model_id,
                       clean_dataset_dir,
                       backdoored_dataset_dir,
                       suffix,
                       'Confusion distributions for a backdoored model',
                       test_ratio,
                       SDNConfig.DenseNet_attach_to_DenseBlocks,
                       device)
    print('script ended')


if __name__ == '__main__':
    main()
