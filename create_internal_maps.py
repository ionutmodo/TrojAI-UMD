import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
import pandas as pd
import tools.aux_funcs as af
from tools.settings import *
from tools.logistics import get_project_root_path
from tools.network_architectures import load_trojai_model
from architectures.SDNs.SDNConfig import SDNConfig
import torch.multiprocessing as mp

###### GLOBAL VARIABLES
sdn_type = SDNConfig.DenseNet_attach_to_DenseBlocks
sdn_name = 'ics_train100_test0_bs25'
cnn_name = 'model.pt'
# device = af.get_pytorch_device()
device = 'cpu'
# locker = mp.Lock()
gaussian_mean = 0.5
gaussian_std = 0.2

def compute_internal_maps(params):
    plots_dir, root_path, noise_path, n_samples_to_use, model_name, num_classes, model_label = params
    # noises = af.load_obj(noise_path)  # method load_obj adds ".pickle" at the end

    if 'train' in os.path.basename(root_path):
        sdn_path = os.path.join(root_path, 'models', model_name)
    else:
        sdn_path = os.path.join(root_path, model_name)

    try:
        global gaussian_mean, gaussian_std, device
        # with torch.no_grad():
        sdn_model = load_trojai_model(sdn_path, sdn_name, cnn_name, num_classes, sdn_type, device)
        sdn_model = sdn_model.eval()
        for i in range(n_samples_to_use):
            # noise_np = noises[np.newaxis, i]
            # noise_np = np.random.uniform(low=0.0, high=1.0, size=TrojAI_input_size).clip(0.0, 1.0)
            noise_np = np.random.normal(loc=gaussian_mean, scale=gaussian_std, size=TrojAI_input_size).clip(0.0, 1.0)
            noise_tt = torch.tensor(noise_np, dtype=torch.float, device=device)

            outputs = sdn_model(noise_tt, include_cnn_out=True)
            softmax_values = []
            logit_values = []
            for logit in outputs:
                soft_max = torch.nn.functional.softmax(logit.to(device), dim=1)
                softmax_values.append(soft_max[0].cpu().detach().numpy())
                logit_values.append(logit[0].cpu().detach().numpy())
            softmax_values = np.array(softmax_values)
            logit_values = np.array(logit_values)

            np.save(f'{plots_dir}/{model_name}_{model_label}_softmax_noise_{i:04d}.npy', softmax_values)
            np.save(f'{plots_dir}/{model_name}_{model_label}_logits_noise_{i:04d}.npy', logit_values)

            # plot_name = f'{model_name}_{model_label}_plot_noise_{i:04d}'
            # plt.imshow(softmax_values)
            # plt.title(plot_name)
            # plt.colorbar()
            # plt.savefig(f'{plots_dir}/{plot_name}.jpg')  # plotting softmax values
            # plt.close()

            del noise_tt, outputs
            torch.cuda.empty_cache()
        del sdn_model
        # print(f'done model {model_name} ({model_label})')
        return True
    except FileNotFoundError:
        return False
    finally:
        sys.stdout.flush()

def main():
    np.random.seed(666)
    project_root_path = get_project_root_path()
    root_path = os.path.join(project_root_path, 'TrojAI-data', 'round1-dataset-train')
    # root_path = os.path.join(project_root_path, 'TrojAI-data', 'round1-holdout-dataset')

    n_samples = 1000
    n_samples_to_use = 100

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    plots_dir = f'internal_maps/{os.path.basename(root_path)}-gaussian-{gaussian_mean:.2f}-{gaussian_std:.2f}'
    af.create_path(plots_dir)

    # noise_images = create_random_normal_noise_images(n_samples, noise_mean, noise_std)
    # af.save_obj(noise_images, os.path.join(plots_dir, f'{plots_dir_basename}-noises-{n_samples}'))
    noise_path = os.path.join(project_root_path,
                              f'TrojAI-UMD',
                              f'confusion_experiments',
                              f'noise_experiments',
                              f'samples-{n_samples}',
                              f'round1-training',
                              f'round1-training-noises-{n_samples}')
    # noises = af.load_obj(noise_path)  # method load_obj adds ".pickle" at the end

    rows = [row for _, row in metadata.iterrows() if row['model_architecture'] == 'densenet121']

    # total_rows = len(rows)
    # for current_row, row in enumerate(rows):
    #     model_name = row['model_name']
    #     num_classes = row['number_classes']
    #     ground_truth = row['ground_truth']
    #     model_label = 'backdoor' if ground_truth else 'clean'
    #
    #     params = (plots_dir, root_path, noise_path, n_samples_to_use, model_name, num_classes, model_label)
    #     status = compute_internal_maps(params)
    #     if status:
    #         print(f'{current_row+1:4d}/{total_rows:4d} done model {model_name} ({model_label})')
    #     else:
    #         print(f'{model_name} does not exist')

    mp.set_start_method('spawn')
    with mp.Pool(processes=4) as pool:
        mapping_params = [
            (plots_dir,
             root_path,
             noise_path,
             n_samples_to_use,
             row['model_name'],
             row['number_classes'],
             'backdoor' if row['ground_truth'] else 'clean')
            for row in rows
        ]
        results = pool.map(compute_internal_maps, mapping_params)


if __name__ == '__main__':
    main()
    print('script ended')

