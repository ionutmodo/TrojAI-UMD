import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

import tools.aux_funcs as af
import tools.model_funcs as mf
from logistics import get_project_root_path
from tools.data import ManualData
from tools.network_architectures import load_trojai_model, get_label_and_confidence_from_logits
from tools.settings import *
from architectures.SDNs.SDNConfig import SDNConfig

def show_label_confidence_for_sdn_outputs(outputs):
    for out in outputs:
        label, confidence = get_label_and_confidence_from_logits(out)
        print(f'label = {label}, confidence = {confidence:.4f}')

def get_random_noise(r_type, r_p1, r_p2):
    noise = None
    if r_type == 'uniform':
        noise = np.random.uniform(low=0.0, high=1.0, size=TrojAI_input_size)
    elif r_type == 'normal':
        noise = np.random.normal(loc=r_p1, scale=r_p2, size=TrojAI_input_size)
    else:
        print('invalid random type! it should be normal or uniform')
    return np.clip(noise, a_min=0.0, a_max=1.0)

def create_random_normal_noise_images(n_samples, param_mean, param_std):
    noise_images = []
    for _ in range(n_samples):
        noise = get_random_noise('normal', param_mean, param_std)
        noise_images.append(noise)
    noise_images = np.array(noise_images).squeeze()
    return noise_images

def label_random_normal_noise_images(data, cnn_model, batch_size):
    device = cnn_model.device
    labels = []

    for i in range(data.shape[0]):
        noise_np = data[i][np.newaxis, :]
        noise_tt = torch.tensor(noise_np, dtype=torch.float, device=device)

        logits = cnn_model(noise_tt)
        label, _ = get_label_and_confidence_from_logits(logits)
        labels.append(label)

    loader = torch.utils.data.DataLoader(ManualData(data, np.array(labels), device),
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4 if device == 'cpu' else 0)
    return loader


# def create_random_noise_dataset_2(n_samples, cnn_clean, cnn_backdoored, batch_size, device, rand_type, rand_p1, rand_p2):
#
#     dataset = []
#     labels_clean = []
#     labels_backdoored = []
#
#     for _ in range(n_samples):
#         noise_np = get_random_noise(r_type=rand_type, r_p1=rand_p1, r_p2=rand_p2)
#         noise_tt = torch.tensor(noise_np, dtype=torch.float, device=device)
#
#         out_clean = cnn_clean(noise_tt)
#         out_backdoored = cnn_backdoored(noise_tt)
#
#         y_clean, _ = get_label_and_confidence_from_logits(out_clean)
#         y_backdoored, _ = get_label_and_confidence_from_logits(out_backdoored)
#
#         dataset.append(noise_np)
#         labels_clean.append(y_clean)
#         labels_backdoored.append(y_backdoored)
#
#     dataset = np.array(dataset).squeeze()
#     labels_clean = np.array(labels_clean)
#     labels_backdoored = np.array(labels_backdoored)
#
#     dataset_clean = ManualData(dataset, labels_clean, device)
#     dataset_backdoored = ManualData(dataset, labels_backdoored, device)
#
#     num_workers = 4 if device == 'cpu' else 0
#     loader_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     loader_backdoored = torch.utils.data.DataLoader(dataset_backdoored, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#     return loader_clean, loader_backdoored

def main():
    np.random.seed(666)
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks
    sdn_name = 'ics_train100_test0_bs25'
    cnn_name = 'model.pt'
    # device = af.get_pytorch_device()
    device = 'cpu'

    n_samples = 10
    batch_size = 50
    noise_mean = 0.5
    noise_std = 0.1

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    stats = pd.DataFrame(columns=['model_id', 'conf_mean', 'conf_std', 'is_backdoored'])
    n_stats = 0

    plots_dir = f'confusion_experiments/noise_experiments/samples-{n_samples}'
    af.create_path(plots_dir)

    clean_model_ids = []
    backdoored_model_ids = []

    id_confusion_dict_clean = {} # key=model_id, value=confusion_scores
    id_loader_dict_clean = {} # key=model_id, value=loader
    id_confusion_dict_backdoored = {} # key=model_id, value=confusion_scores
    id_loader_dict_backdoored = {} # key=model_id, value=loader

    noise_images = create_random_normal_noise_images(n_samples, noise_mean, noise_std)

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        model_architecture = row['model_architecture']
        # num_classes = row['number_classes']
        ground_truth = row['ground_truth']

        if model_architecture == 'densenet121':
            if ground_truth:
                backdoored_model_ids.append(model_name)
            else:
                clean_model_ids.append(model_name)

    for id_clean in clean_model_ids:
        print('clean model', id_clean)
        sdn_path_clean = os.path.join(root_path, id_clean)
        sdn_clean = load_trojai_model(sdn_path_clean, sdn_name, cnn_name, TrojAI_num_classes, sdn_type, device)
        loader_clean = label_random_normal_noise_images(noise_images, sdn_clean.model, batch_size)

        clean_confusion_scores = mf.compute_confusion(sdn_clean, loader_clean, device)
        clean_mean = clean_confusion_scores.mean()
        clean_std = clean_confusion_scores.std()

        id_confusion_dict_clean[id_clean] = clean_confusion_scores
        id_loader_dict_clean[id_clean] = loader_clean

        stats.loc[n_stats] = [id_clean, clean_mean, clean_std, 0]
        n_stats += 1
    print()
    for id_backdoored in backdoored_model_ids:
        print('backdoored model', id_backdoored)
        sdn_path_backdoored = os.path.join(root_path, id_backdoored)
        sdn_backdoored = load_trojai_model(sdn_path_backdoored, sdn_name, cnn_name, TrojAI_num_classes, sdn_type, device)
        loader_backdoored = label_random_normal_noise_images(noise_images, sdn_backdoored.model, batch_size)

        backdoored_confusion_scores = mf.compute_confusion(sdn_backdoored, loader_backdoored, device)
        backdoored_mean = backdoored_confusion_scores.mean()
        backdoored_std = backdoored_confusion_scores.std()

        id_confusion_dict_backdoored[id_backdoored] = backdoored_confusion_scores
        id_loader_dict_backdoored[id_backdoored] = loader_backdoored

        stats.loc[n_stats] = [id_backdoored, backdoored_mean, backdoored_std, 1]
        n_stats += 1
    print()
    for id_clean in clean_model_ids:
        clean_confusion_scores = id_confusion_dict_clean[id_clean]
        clean_mean = clean_confusion_scores.mean()
        clean_std = clean_confusion_scores.std()
        for id_backdoored in backdoored_model_ids:
            backdoored_confusion_scores = id_confusion_dict_backdoored[id_backdoored]
            backdoored_mean = backdoored_confusion_scores.mean()
            backdoored_std = backdoored_confusion_scores.std()

            conf_mean_diff = abs(clean_confusion_scores.mean() - backdoored_confusion_scores.mean())

            save_name = f'noised_normal-{noise_mean:.2f}-{noise_std:.2f}_datasets_samples-{n_samples}_ids-{id_clean}-{id_backdoored}.png'
            title = f'N(mean={noise_mean:.2f}, std={noise_std:.2f}) pixels\n' \
                    f'clean conf dist: m={clean_mean:.2f}, s={clean_std:.2f}\n' \
                    f'backd conf dist: m={backdoored_mean:.2f}, s={backdoored_std:.2f}\n' \
                    f'conf dist means difference = {conf_mean_diff:.2f}'

            af.overlay_two_histograms(save_path=plots_dir,
                                      save_name=save_name,
                                      hist_first_values=clean_confusion_scores,
                                      hist_second_values=backdoored_confusion_scores,
                                      first_label='clean model',
                                      second_label='backdoored model',
                                      xlabel='Confusion score',
                                      title=title)

    id_confusion_dict_clean = {} # key=model_id, value=confusion_scores
    id_loader_dict_clean = {} # key=model_id, value=loader
    id_confusion_dict_backdoored = {} # key=model_id, value=confusion_scores
    id_loader_dict_backdoored = {} # key=model_id, value=loader

    af.save_obj(id_confusion_dict_clean,      os.path.join(plots_dir, f'confusion_dict_clean-{n_samples}'))
    af.save_obj(id_loader_dict_clean,         os.path.join(plots_dir, f'loader_dict_clean-{n_samples}'))
    af.save_obj(id_confusion_dict_backdoored, os.path.join(plots_dir, f'confusion_dict_backdoored-{n_samples}'))
    af.save_obj(id_loader_dict_backdoored,    os.path.join(plots_dir, f'loader_dict_backdoored-{n_samples}'))

    stats.to_csv(os.path.join(plots_dir, f'stats-{n_samples}.csv'), index=False)


if __name__ == '__main__':
    main()
    print('script ended')
