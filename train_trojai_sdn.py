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
from tools.logger import Logger

from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.MLP import LayerwiseClassifiers


def train_trojai_sdn(dataset, trojai_model_w_ics, model_root_path, device):
    output_params = trojai_model_w_ics.get_layerwise_model_params()

    # mlp_num_layers = 2
    # mlp_architecture_param =  [2, 2] # use [2] if it takes too much time
    
    mlp_num_layers = 0
    mlp_architecture_param = [] # empty architecture param means that the MLP won't have any hidden layers, it will be a linear perceptron
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


def train_trojai_sdn_with_svm(dataset, trojai_model_w_ics, model_root_path, device, log=False):
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

        svm_ics.append(svm)
        if log:
            y_pred = svm.predict(features[i])

            list_raw_acc = []
            list_bal_acc = []
            for c in range(n_classes):
                acc_raw = accuracy_score(y_true=labels[:, c], y_pred=y_pred[:, c])
                acc_balanced = balanced_accuracy_score(y_true=labels[:, c], y_pred=y_pred[:, c])

                list_raw_acc.append(f'{acc_raw * 100.0:.2f}')
                list_bal_acc.append(f'{acc_balanced * 100.0:.2f}')

            Logger.log(f'SVM-IC-{i} Raw Acc: [{", ".join(list_raw_acc)}]')
            Logger.log(f'SVM-IC-{i} Bal Acc: [{", ".join(list_bal_acc)}]')
            Logger.log(f'--------------------------------------------------------------------------')

    path_svm = os.path.join(model_root_path, 'svm')
    af.create_path(path_svm)

    path_svm_model = os.path.join(path_svm, 'svm_models')
    af.save_obj(obj=svm_ics, filename=path_svm_model)
    size = os.path.getsize(path_svm_model) / (2 ** 20)
    if log:
        Logger.log(f'SVM model ({size:.2f} MB) saved to {path_svm_model}')

    path_svm_dataset = os.path.join(path_svm, 'svm_dataset')
    af.save_obj(obj={'features': features, 'labels': labels}, filename=path_svm_dataset)
    size = os.path.getsize(path_svm_dataset) / (2 ** 20)
    if log:
        Logger.log(f'SVM dataset ({size:.2f} MB) saved to {path_svm_dataset}')


def main():
    af.set_random_seeds()

    device = af.get_pytorch_device()
    # device = 'cpu'

    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-dataset-train')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-train-dataset')

    path_logger = os.path.join(root_path, f'{os.path.basename(root_path)}.log')
    Logger.open(path_logger)

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    if 'train' in os.path.basename(root_path) and 'round1' in os.path.basename(root_path): # append 'models' for training dataset
        root_path = os.path.join(root_path, 'models')

    batch_size = 1
    test_ratio = 0

    dict_arch_type = {
        'densenet': SDNConfig.DenseNet_blocks,
        'googlenet': SDNConfig.GoogLeNet,
        'inception': SDNConfig.Inception3,
        'mobilenet': SDNConfig.MobileNet2,
        'resnet': SDNConfig.ResNet,
        'shufflenet': SDNConfig.ShuffleNet,
        'squeezenet': SDNConfig.SqueezeNet,
        'vgg': SDNConfig.VGG,
        'wideresnet': SDNConfig.ResNet,
    }

    Logger.log('!!! Round2: opencv_format=False in TrojAI constructor!!!')

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        # model_id = int(model_name[3:])
        model_architecture = row['model_architecture']
        num_classes = row['number_classes']
        poisoned = 'backdoored' if bool(row['poisoned']) else 'clean'

        for arch_prefix, sdn_type in dict_arch_type.items():
            if model_architecture.startswith(arch_prefix):
                model_root = os.path.join(root_path, model_name)
                Logger.log(f'Training {model_architecture}-sdn ({poisoned}) in {model_root}')

                time_start = datetime.now()

                dataset, model = read_model_directory(model_root, num_classes, batch_size, test_ratio, sdn_type, device)
                # train_trojai_sdn(dataset, model, model_root, device)
                train_trojai_sdn_with_svm(dataset, model, model_root, device)

                time_end = datetime.now()
                Logger.log(f'elapsed {time_end - time_start}\n')
    Logger.log('script ended')
    Logger.close()


if __name__ == "__main__":
    main()
