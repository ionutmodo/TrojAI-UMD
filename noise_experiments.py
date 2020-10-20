import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy import stats

import tools.aux_funcs as af
import tools.model_funcs as mf
from tools.logistics import get_project_root_path
from tools.data import ManualData
from tools.network_architectures import load_trojai_model, get_label_and_confidence_from_logits
from tools.settings import *
from SDNConfig import SDNConfig

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
    labels = np.array(labels)
    return labels

def create_loader(data, labels, batch_size, device):
    dataset = ManualData(data, labels, device)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4 if device == 'cpu' else 0)
    return loader

def main():
    np.random.seed(666)

    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-dataset-train')
    sdn_type = SDNConfig.DenseNet_blocks
    sdn_name = 'ics_train100_test0_bs25'
    cnn_name = 'model.pt'
    device = af.get_pytorch_device()
    # device = 'cpu'

    n_samples = 1000
    batch_size = 25
    noise_mean = 0.5
    noise_std = 0.1

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    # plots_dir = f'confusion_experiments/noise_experiments/samples-{n_samples}/round1-training'
    plots_dir = f'confusion_experiments/noise_experiments/samples-{n_samples}/round1-holdout'
    plots_dir_basename = os.path.basename(plots_dir)
    af.create_path(plots_dir)

    model_ids_clean = []
    model_ids_backdoored = []

    dict_id_confusion = {}
    dict_id_labels = {}

    # noise_images = create_random_normal_noise_images(n_samples, noise_mean, noise_std)
    # af.save_obj(noise_images, os.path.join(plots_dir, f'{plots_dir_basename}-noises-{n_samples}'))
    noise_images = af.load_obj(os.path.join(get_project_root_path(),
                                            f'TrojAI-UMD',
                                            f'confusion_experiments',
                                            f'noise_experiments',
                                            f'samples-{n_samples}',
                                            f'round1-training',
                                            f'round1-training-noises-{n_samples}')) # method load_obj adds ".pickle" at the end

    stats_df = pd.DataFrame(columns=['id', 'mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max', 'ground_truth'])
    n_stats = 0

    total_models = len(metadata[metadata['model_architecture'] == 'densenet121'])

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        model_architecture = row['model_architecture']
        num_classes = row['number_classes']
        ground_truth = row['ground_truth']

        if model_architecture == 'densenet121': # and int(model_name[3:]) in already_trained_model_ids:
            # sdn_path = os.path.join(root_path, model_name)

            # just for debugging purposes
            # model_name = 'id-00000004'
            # sdn_path = os.path.join(root_path, model_name)

            if 'training' in os.path.basename(root_path):
                sdn_path = os.path.join(root_path, 'models', model_name)
            else:
                sdn_path = os.path.join(root_path, model_name)

            sdn_model = load_trojai_model(sdn_path, sdn_name, cnn_name, num_classes, sdn_type, device)
            labels = label_random_normal_noise_images(noise_images, sdn_model.cnn_model, batch_size)
            loader = create_loader(noise_images, labels, batch_size, device)

            confusion_scores = mf.compute_confusion(sdn_model, loader, device)

            dict_id_confusion[model_name] = confusion_scores
            dict_id_labels[model_name] = labels

            stat_mean = np.mean(confusion_scores)
            stat_std = np.std(confusion_scores)
            stat_median = np.median(confusion_scores)
            stat_skewness = stats.skew(confusion_scores)
            stat_kurtosis = stats.kurtosis(confusion_scores)
            stat_min = np.min(confusion_scores)
            stat_max = np.max(confusion_scores)

            stats_df.loc[n_stats] = [model_name, stat_mean, stat_std, stat_median, stat_skewness, stat_kurtosis, stat_min, stat_max, int(ground_truth)]
            n_stats += 1

            if ground_truth:
                model_ids_backdoored.append(model_name)
            else:
                model_ids_clean.append(model_name)
            print(f'{n_stats:4d}/{total_models:4d} done model {model_name} ({"backdoored" if ground_truth else "clean"})')
            sys.stdout.flush()

    stats_df = stats_df.set_index('id')

    # for id_clean in model_ids_clean:
    #     clean_confusion_scores = dict_id_confusion[id_clean]
    #
    #     clean_mean = stats_df.loc[id_clean, 'mean']
    #     clean_std = stats_df.loc[id_clean, 'std']
    #     clean_median = stats_df.loc[id_clean, 'median']
    #     clean_skewness = stats_df.loc[id_clean, 'skewness']
    #     clean_kurtosis = stats_df.loc[id_clean, 'kurtosis']
    #
    #     for id_backdoored in model_ids_backdoored:
    #         backdoored_confusion_scores = dict_id_confusion[id_backdoored]
    #
    #         backdoored_mean = stats_df.loc[id_backdoored, 'mean']
    #         backdoored_std = stats_df.loc[id_backdoored, 'std']
    #         backdoored_median = stats_df.loc[id_backdoored, 'median']
    #         backdoored_skewness = stats_df.loc[id_backdoored, 'skewness']
    #         backdoored_kurtosis = stats_df.loc[id_backdoored, 'kurtosis']
    #
    #         save_name = f'noised_normal-{noise_mean:.2f}-{noise_std:.2f}_datasets_samples-{n_samples}_ids-{id_clean}-{id_backdoored}.png'
    #         title = f'N(mean={noise_mean:.2f}, std={noise_std:.2f}) pixels\n' \
    #                 f'clean: m={clean_mean:.2f},sd={clean_std:.2f},M={clean_median:.2f},sk={clean_skewness:.2f},k={clean_kurtosis:.2f}\n' \
    #                 f'backd: m={backdoored_mean:.2f},sd={backdoored_std:.2f},M={backdoored_median:.2f},sk={backdoored_skewness:.2f},k={backdoored_kurtosis:.2f}\n' \
    #
    #         af.overlay_two_histograms(save_path=plots_dir,
    #                                   save_name=save_name,
    #                                   hist_first_values=clean_confusion_scores,
    #                                   hist_second_values=backdoored_confusion_scores,
    #                                   first_label='clean model',
    #                                   second_label='backdoored model',
    #                                   xlabel='Confusion score',
    #                                   title=title)

    af.save_obj(dict_id_confusion, os.path.join(plots_dir, f'{plots_dir_basename}-confusion-{n_samples}'))
    af.save_obj(dict_id_labels,    os.path.join(plots_dir, f'{plots_dir_basename}-labels-{n_samples}'))

    stats_df.to_csv(os.path.join(plots_dir, f'{plots_dir_basename}-stats-{n_samples}.csv'), index=True)


if __name__ == '__main__':
    main()
    print('script ended')


# def label_random_normal_noise_images_batch(data, cnn_model, batch_size):
#     device = cnn_model.device
#     labels = []
#     n_samples = data.shape[0]
#     n_batches = int(n_samples / batch_size) + 1
#     for i in range(n_batches):
#         left = i * batch_size
#         right = min(n_samples, (i + 1) * batch_size)
#         noise_np = data[left:right]
#         noise_tt = torch.tensor(noise_np, dtype=torch.float, device=device)
#
#         logits = cnn_model(noise_tt)
#         # for logit in logits:
#         for j in range(right-left):
#             logit = torch.unsqueeze(logits[j], dim=0) # size=5 becomes size=[1, 5]
#             label, _ = get_label_and_confidence_from_logits(logit)
#             labels.append(label)
#     labels = np.array(labels)
#     return labels