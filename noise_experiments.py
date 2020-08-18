import os
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

def main():
    np.random.seed(666)
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks
    sdn_name = 'ics_train100_test0_bs25'
    cnn_name = 'model.pt'
    device = af.get_pytorch_device()
    # device = 'cpu'

    n_samples = 1000
    batch_size = 50
    noise_mean = 0.5
    noise_std = 0.1

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    plots_dir = f'confusion_experiments/noise_experiments/samples-{n_samples}'
    af.create_path(plots_dir)

    model_ids_clean = []
    model_ids_backdoored = []

    dict_id_confusion = {}
    dict_id_loader = {}

    noise_images = create_random_normal_noise_images(n_samples, noise_mean, noise_std)

    stats_df = pd.DataFrame(columns=['id', 'mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max', 'ground_truth'])
    n_stats = 0

    # clean_model_ids = [4, 7, 25, 27, 40]
    # backdoored_model_ids = [2, 9, 13, 24, 26]
    # already_trained_model_ids = clean_model_ids + backdoored_model_ids

    total_models = len(metadata[metadata['model_architecture'] == 'densenet121'])

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        model_architecture = row['model_architecture']
        num_classes = row['number_classes']
        ground_truth = row['ground_truth']

        if model_architecture == 'densenet121': # and int(model_name[3:]) in already_trained_model_ids:
            sdn_path = os.path.join(root_path, model_name)
            sdn_model = load_trojai_model(sdn_path, sdn_name, cnn_name, num_classes, sdn_type, device)

            loader = label_random_normal_noise_images(noise_images, sdn_model.model, batch_size)
            confusion_scores = mf.compute_confusion(sdn_model, loader, device)

            dict_id_confusion[model_name] = confusion_scores
            dict_id_loader[model_name] = loader

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

    stats_df = stats_df.set_index('id')

    for id_clean in model_ids_clean:
        clean_confusion_scores = dict_id_confusion[id_clean]

        clean_mean = stats_df.loc[id_clean, 'mean']
        clean_std = stats_df.loc[id_clean, 'std']
        clean_median = stats_df.loc[id_clean, 'median']
        clean_skewness = stats_df.loc[id_clean, 'skewness']
        clean_kurtosis = stats_df.loc[id_clean, 'kurtosis']

        for id_backdoored in model_ids_backdoored:
            backdoored_confusion_scores = dict_id_confusion[id_backdoored]

            backdoored_mean = stats_df.loc[id_backdoored, 'mean']
            backdoored_std = stats_df.loc[id_backdoored, 'std']
            backdoored_median = stats_df.loc[id_backdoored, 'median']
            backdoored_skewness = stats_df.loc[id_backdoored, 'skewness']
            backdoored_kurtosis = stats_df.loc[id_backdoored, 'kurtosis']

            save_name = f'noised_normal-{noise_mean:.2f}-{noise_std:.2f}_datasets_samples-{n_samples}_ids-{id_clean}-{id_backdoored}.png'
            title = f'N(mean={noise_mean:.2f}, std={noise_std:.2f}) pixels\n' \
                    f'clean: m={clean_mean:.2f},sd={clean_std:.2f},M={clean_median:.2f},sk={clean_skewness:.2f},k={clean_kurtosis:.2f}\n' \
                    f'backd: m={backdoored_mean:.2f},sd={backdoored_std:.2f},M={backdoored_median:.2f},sk={backdoored_skewness:.2f},k={backdoored_kurtosis:.2f}\n' \

            af.overlay_two_histograms(save_path=plots_dir,
                                      save_name=save_name,
                                      hist_first_values=clean_confusion_scores,
                                      hist_second_values=backdoored_confusion_scores,
                                      first_label='clean model',
                                      second_label='backdoored model',
                                      xlabel='Confusion score',
                                      title=title)

    af.save_obj(dict_id_confusion, os.path.join(plots_dir, f'confusion-{n_samples}'))
    af.save_obj(dict_id_loader,    os.path.join(plots_dir, f'loader-{n_samples}'))

    stats_df.to_csv(os.path.join(plots_dir, f'stats-{n_samples}.csv'), index=True)


if __name__ == '__main__':
    main()
    print('script ended')
