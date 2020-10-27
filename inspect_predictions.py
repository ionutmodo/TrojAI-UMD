from data import TrojAI
from scipy.stats import entropy
import numpy as np
import pandas as pd
import torch
import os
from tools.logistics import get_project_root_path


def get_predicted_label(model, image, device):
    output = model(image.to(device))
    # softmax = nn.functional.softmax(output[0].cpu(), dim=0)
    pred_label = output.max(1)[1].item()
    return pred_label


def get_confusion_matrix(model, path_dataset, device):
    dataset = TrojAI(folder=path_dataset, test_ratio=0, batch_size=1, device=device, opencv_format=False)

    matrix_labels = [[0] * dataset.num_classes for _ in range(dataset.num_classes)]
    for image, label_true in dataset.train_loader:
        label_pred = get_predicted_label(model, image, device)
        matrix_labels[label_true.item()][label_pred] += 1
    return matrix_labels


def print_confusion_matrix(message, matrix):
    print(message)
    for i, L in enumerate(matrix):
        print(f'{i:2d}', L)
    matrix = np.array(matrix)
    column_mean = matrix.mean(axis=0)
    proba = column_mean / column_mean.sum()
    print('mean', [f'{x:.02f}' for x in column_mean.tolist()])
    print('prob', [f'{x:.02f}' for x in proba.tolist()])
    print(f' H={entropy(proba):.010f}')
    uniform = np.ones_like(proba) / proba.shape[0]
    print(f'KL(p,u)={entropy(proba, uniform):.010f}')
    print(f'KL(u,p)={entropy(uniform, proba):.010f}')
    print()


def main():
    device = 'cuda'

    path_metadata = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-train-dataset', 'METADATA.csv')
    metadata = pd.read_csv(path_metadata)

    for model_id in ['id-00000000', 'id-00000001', 'id-00000002', 'id-00000007', 'id-00000023', 'id-00000024', 'id-00000025', 'id-00000027', 'id-00000028', 'id-00000047']:
        row = metadata[metadata['model_name'] == model_id].iloc[0]

        path_model = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\model.pt'
        model = torch.load(path_model, map_location=device).eval()

        path_data_clean = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\example_data'
        path_data_polygon = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\backdoored_data_custom-square-size-25_backd-original-color_clean-black-color'
        path_data_gotham = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\backdoored_data_filter_gotham'
        path_data_kelvin = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\backdoored_data_filter_kelvin'
        path_data_lomo = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\backdoored_data_filter_lomo'
        path_data_nashville = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\backdoored_data_filter_nashville'
        path_data_toaster = rf'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\{model_id}\backdoored_data_filter_toaster'

        message = f'{model_id}-{str(row["poisoned"])}-{str(row["trigger_type_option"])}-{str(row["triggered_classes"])}-{str(row["trigger_target_class"])}'

        print_confusion_matrix(f'clean-{message}', get_confusion_matrix(model, path_data_clean, device))
        print_confusion_matrix(f'polygon-{message}', get_confusion_matrix(model, path_data_polygon, device))
        print_confusion_matrix(f'gotham-{message}', get_confusion_matrix(model, path_data_gotham, device))
        print_confusion_matrix(f'kelvin-{message}', get_confusion_matrix(model, path_data_kelvin, device))
        print_confusion_matrix(f'lomo-{message}', get_confusion_matrix(model, path_data_lomo, device))
        print_confusion_matrix(f'nashville-{message}', get_confusion_matrix(model, path_data_nashville, device))
        print_confusion_matrix(f'toaster-{message}', get_confusion_matrix(model, path_data_toaster, device))
        print('-----------------------------------------------------------------')


if __name__ == '__main__':
    main()
