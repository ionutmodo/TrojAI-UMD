import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from datetime import datetime
import tools.aux_funcs as af
import tools.model_funcs as mf
import tools.network_architectures as arcs
from tools.logistics import *

from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.MLP import LayerwiseClassifiers


def train_trojai_sdn(dataset, trojai_model_w_ics, model_root_path, device):
    output_params = trojai_model_w_ics.get_layerwise_model_params()

    mlp_num_layers = 2
    mlp_architecture_param = [2, 2] # use [2] if it takes too much time
    # think about simplifying ICs architecture
    architecture_params = (mlp_num_layers, mlp_architecture_param)

    params = {
        'network_type': 'layerwise_classifiers',
        'output_params': output_params,
        'architecture_params': architecture_params
    }

    # settings to train the MLPs
    epochs = 20
    lr_params = (0.001, 0.00001)
    stepsize_params = ([10], [0.1])

    sys.stdout.flush()
    ics = LayerwiseClassifiers(output_params, architecture_params).to(device)
    ics.set_model(trojai_model_w_ics)

    optimizer, scheduler = af.get_optimizer(ics, lr_params, stepsize_params, optimizer='adam')

    print(f'Training SDN version for model {os.path.basename(model_root_path)}')
    sys.stdout.flush()
    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    test_proc = int(dataset.test_ratio * 100)
    train_proc = 100 - test_proc
    bs = dataset.batch_size
    arcs.save_model(ics, params, model_root_path, f'ics_train{train_proc}_test{test_proc}_bs{bs}', epoch=-1)


def train_trojai_sdn_with_svm(dataset, trojai_model_w_ics, model_root_path, device):
    ic_count = len(trojai_model_w_ics.get_layerwise_model_params())
    trojai_model_w_ics.eval().to(device)

    # list to save the features for each IC
    # features[i] = the dataset to train the SVM for IC_i
    features = [[] for _ in range(ic_count)]
    labels = []

    for batch_x, batch_y in dataset.train_loader:
        activations, out = trojai_model_w_ics.forward_w_acts(batch_x)
        for i, act in enumerate(activations):
            features[i].append(act[0].cpu().detach().numpy())
        for y in batch_y:
            labels.append(y.item())

    classes = list(set(labels))
    n_classes = len(classes)
    labels = label_binarize(labels, classes=sorted(classes))

    for i in range(ic_count):
        features[i] = np.array(features[i])

    svm_ics = []
    for i in range(ic_count):
        svm = OneVsRestClassifier(estimator=SVC(kernel='linear', probability=True, random_state=0),
                                  n_jobs=n_classes)
        svm.fit(features[i], labels)
        y_pred = svm.predict(features[i])

        acc_raw = accuracy_score(y_true=labels[:, i], y_pred=y_pred[:, i])
        acc_balanced = balanced_accuracy_score(y_true=labels[:, i], y_pred=y_pred[:, i])

        print(f'SVM-IC-{i}: raw accuracy={acc_raw}, balanced accuracy={acc_balanced}')
        svm_ics.append(svm)

    path_svm = os.path.join(model_root_path, 'ics_svm')
    af.save_obj(svm_ics, path_svm)
    size = os.path.getsize(path_svm) / (2 ** 20)
    print(f'SVM ICs has {size:.2f} MB and was saved to {path_svm}\n')


def main():
    af.set_random_seeds()

    device = af.get_pytorch_device()
    # device = 'cpu'

    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-dataset-train')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-train-dataset')

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    if 'train' in os.path.basename(root_path) and 'round1' in os.path.basename(root_path): # append 'models' for training dataset
        root_path = os.path.join(root_path, 'models')

    batch_size = 1
    test_ratio = 0

    dict_arch_type = {
        'densenet': SDNConfig.DenseNet_attach_to_DenseBlocks,
        'resnet': SDNConfig.ResNet50,
        'inceptionv3': SDNConfig.Inception3
    }
    # sdn_type, architecture_to_train = SDNConfig.DenseNet_attach_to_DenseBlocks, 'densenet121'
    # sdn_type, architecture_to_train = SDNConfig.ResNet50, 'resnet50'
    # sdn_type, architecture_to_train = SDNConfig.Inception3, 'inceptionv3'

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        model_architecture = row['model_architecture']
        num_classes = row['number_classes']

        for arch_prefix, sdn_type in dict_arch_type.items():
            if model_architecture.startswith(arch_prefix):
                model_root = os.path.join(root_path, model_name)
                print(f'Training {model_architecture}-sdn in {model_root}')

                time_start = datetime.now()

                dataset, model_label, model = read_model_directory(model_root, num_classes, batch_size, test_ratio, sdn_type, device)
                # train_trojai_sdn(dataset, model, model_root, device)
                train_trojai_sdn_with_svm(dataset, model, model_root, device)

                time_end = datetime.now()
                print(f'elapsed {time_end - time_start}')
    print('script ended')


if __name__ == "__main__":
    main()
