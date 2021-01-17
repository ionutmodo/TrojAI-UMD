from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tools.network_architectures import save_model
from architectures.MLP import LayerwiseClassifiers
from tools.logger import Logger
import tools.model_funcs as mf
import tools.aux_funcs as af
import numpy as np
import sys
import os


def encode_architecture(model_architecture):
    arch_codes = ['densenet', 'googlenet', 'inception', 'mobilenet', 'resnet', 'shufflenet', 'squeezenet', 'vgg']
    for index, arch in enumerate(arch_codes):
        if arch in model_architecture:
            return index
    return None


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
            # add a for loop here to save all features in the batch act[0..n-1, :]
            features[i].append(act[0].cpu().detach().numpy())
        for y in batch_y:
            labels.append(y.item())

    classes = list(set(labels))
    n_classes = len(classes)
    labels = label_binarize(labels, classes=sorted(classes))

    print(f'[info] number of classes: {n_classes}')
    for i in range(ic_count):
        list_size = len(features[i])
        item_size = len(features[i][0])
        features[i] = np.array(features[i])
        print(f'[info] shape for IC#{i}: {features[i].shape}, list_size={list_size}, item_size={item_size}')

    svm_ics = []
    for i in range(ic_count):
        svm = OneVsRestClassifier(estimator=SVC(kernel='linear', probability=True, random_state=0), n_jobs=n_classes)
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

            print(f'[info] SVM-IC-{i} Raw Acc: [{", ".join(list_raw_acc)}]')
            print(f'[info] SVM-IC-{i} Bal Acc: [{", ".join(list_bal_acc)}]')
            print(f'--------------------------------------------------------------------------')

    path_svm_model = os.path.join(model_root_path, 'ics_svm.model')
    af.save_obj(obj=svm_ics, filename=path_svm_model)
    size = os.path.getsize(path_svm_model) / (2 ** 20)
    if log:
        Logger.log(f'[info] SVM model ({size:.2f} MB) saved to {path_svm_model}')


def train_trojai_sdn_with_fc(dataset, trojai_model_w_ics, model_root_path, device):
    output_params = trojai_model_w_ics.get_layerwise_model_params()

    # mlp_num_layers = 2
    # mlp_architecture_param =  [2, 2] # use [2] if it takes too much time

    mlp_num_layers = 0
    mlp_architecture_param = []  # empty architecture param means that the MLP won't have any hidden layers, it will be a linear perceptron
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

    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    test_proc = int(dataset.test_ratio * 100)
    train_proc = 100 - test_proc
    bs = dataset.batch_size
    ics_model_name = f'ics_synthetic-1000_train{train_proc}_test{test_proc}_bs{bs}'
    save_model(ics, params, model_root_path, ics_model_name, epoch=-1)
