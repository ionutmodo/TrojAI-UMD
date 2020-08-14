import os
import numpy as np
import torch

import tools.aux_funcs as af
import tools.model_funcs as mf
from tools.data import ManualData
from tools.network_architectures import load_trojai_model, get_label_and_confidence_from_logits
from tools.settings import *
from architectures.SDNs.SDNConfig import SDNConfig

def show_label_confidence_for_sdn_outputs(outputs):
    for out in outputs:
        label, confidence = get_label_and_confidence_from_logits(out)
        print(f'label = {label}, confidence = {confidence:.4f}')


def create_random_noise_dataset(n_samples, cnn_clean, cnn_backdoored, batch_size, device, rand_type, rand_p1, rand_p2):
    def get_random_noise(r_type, r_p1, r_p2):
        noise = None
        if r_type == 'uniform':
            noise = np.random.uniform(low=r_p1, high=r_p2, size=TrojAI_input_size)
        elif r_type == 'normal':
            noise = np.random.normal(loc=r_p1, scale=r_p2, size=TrojAI_input_size)
        else:
            print('invalid random type! it should be normal or uniform')
        return np.clip(noise, a_min=0.0, a_max=1.0)
    dataset = []
    labels_clean = []
    labels_backdoored = []

    for _ in range(n_samples):
        noise_np = get_random_noise('normal', r_p1=rand_p1, r_p2=rand_p2)
        noise_tt = torch.tensor(noise_np, dtype=torch.float, device=device)

        out_clean = cnn_clean(noise_tt)
        out_backdoored = cnn_backdoored(noise_tt)

        y_clean, _ = get_label_and_confidence_from_logits(out_clean)
        y_backdoored, _ = get_label_and_confidence_from_logits(out_backdoored)

        dataset.append(noise_np)
        labels_clean.append(y_clean)
        labels_backdoored.append(y_backdoored)

    dataset = np.array(dataset).squeeze()
    labels_clean = np.array(labels_clean)
    labels_backdoored = np.array(labels_backdoored)

    dataset_clean = ManualData(dataset, labels_clean, device)
    dataset_backdoored = ManualData(dataset, labels_backdoored, device)

    num_workers = 4 if device == 'cpu' else 0
    loader_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_backdoored = torch.utils.data.DataLoader(dataset_backdoored, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader_clean, loader_backdoored


def main():
    np.random.seed(666)
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks
    sdn_name = 'ics_train100_test0_bs25'
    cnn_name = 'model.pt'
    # device = af.get_pytorch_device()
    device = 'cpu'

    id_clean, id_backdoored = 4, 9
    n_samples = 500
    batch_size = 250
    rand_type = 'normal'

    sdn_path_clean = os.path.join(root_path, f'id-{id_clean:08d}')
    sdn_path_backdoored = os.path.join(root_path, f'id-{id_backdoored:08d}')

    sdn_clean      = load_trojai_model(sdn_path_clean,      sdn_name, cnn_name, TrojAI_num_classes, sdn_type, device)
    sdn_backdoored = load_trojai_model(sdn_path_backdoored, sdn_name, cnn_name, TrojAI_num_classes, sdn_type, device)

    for rand_p1 in [0.5]:
        for rand_p2 in [0.1, 0.25, 0.5, 0.75, 1.0]:
            print(f'p1={rand_p1}, p2={rand_p2}')
            loader_clean, loader_backdoored = create_random_noise_dataset(n_samples, sdn_clean.model, sdn_backdoored.model, batch_size, device, rand_type,
                                                                          rand_p1, rand_p2)

            # clean_mean, clean_std = mf.sdn_confusion_stats(sdn_clean, loader=loader_clean, device=device)
            # print(f'clean confusion: mean={clean_mean}, std={clean_std}')

            clean_confusion_scores = mf.compute_confusion(sdn_clean, loader_clean, device)
            print(f'clean confusion: mean={clean_confusion_scores.mean()}, std={clean_confusion_scores.std()}')
            # clean_confusion_scores = (clean_confusion_scores - clean_mean) / clean_std

            backdoored_confusion_scores = mf.compute_confusion(sdn_backdoored, loader_backdoored, device)
            print(f'backdoored confusion: mean={backdoored_confusion_scores.mean()}, std={backdoored_confusion_scores.std()}')
            # backdoored_confusion_scores = (backdoored_confusion_scores - clean_mean) / clean_std  # divide backdoored by clean mean/std!

            plots_dir = f'confusion_experiments/noise_experiments'
            af.create_path(plots_dir)

            conf_mean_diff = abs(clean_confusion_scores.mean() - backdoored_confusion_scores.mean())

            if rand_type == 'uniform':
                save_name = f'noised_uniform_datasets_samples-{n_samples}_ids-{id_clean}-{id_backdoored}.png'
                title = f'U[0,1) pixels\nmeans difference = {conf_mean_diff:.2f}'
            else:
                save_name = f'noised_normal-{rand_p1:.2f}-{rand_p2:.2f}_datasets_samples-{n_samples}_ids-{id_clean}-{id_backdoored}.png'
                title = f'N(mean={rand_p1:.2f}, std={rand_p2:.2f}) pixels\nmeans difference = {conf_mean_diff:.2f}'

            af.overlay_two_histograms(save_path=plots_dir,
                                      save_name=save_name,
                                      hist_first_values=clean_confusion_scores,
                                      hist_second_values=backdoored_confusion_scores,
                                      first_label='clean model',
                                      second_label='backdoored model',
                                      xlabel='Confusion score',
                                      title=title)
            print()

if __name__ == '__main__':
    main()
    print('script ended')
