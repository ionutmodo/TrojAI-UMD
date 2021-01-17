import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from datetime import datetime
import tools.model_funcs as mf
import tools.network_architectures as arcs
from tools.logistics import *
from tools.logger import Logger
from architectures.SDNConfig import SDNConfig
from architectures.MLP import LayerwiseClassifiers
import synthetic_data.gen_backdoored_datasets as synthetic_module
import synthetic_data.aux_funcs as sdaf

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

    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    test_proc = int(dataset.test_ratio * 100)
    train_proc = 100 - test_proc
    bs = dataset.batch_size
    ics_model_name = f'ics_synthetic-1000_train{train_proc}_test{test_proc}_bs{bs}'
    arcs.save_model(ics, params, model_root_path, ics_model_name, epoch=-1)


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
    lim_left, lim_right = 0, 1007
    if len(sys.argv) == 3:
        lim_left, lim_right = int(sys.argv[1]), int(sys.argv[2])

    af.set_random_seeds()

    device = af.get_pytorch_device()
    # device = 'cpu'

    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-dataset-train')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-train-dataset')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-holdout-dataset')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round3-train-dataset')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round3-holdout-dataset')
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round4-train-dataset')

    path_logger = os.path.join(root_path, f'{os.path.basename(root_path)}_{lim_left}-{lim_right}.log')
    Logger.open(path_logger)

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    batch_size = 20 # set to 1 for SVM-based ICs
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

    Logger.log(f'lim_left={lim_left}, lim_right={lim_right}')

    ############################################
    ########## LOAD SYNTHETIC DATASET ##########
    synthetic_data = np.load('synthetic_data/synthetic_data_1000_clean_polygon_instagram.npz')

    for index, row in metadata.iterrows():
        model_name = row['model_name']
        model_id = int(model_name[3:])
        if lim_left <= model_id <= lim_right:
            model_architecture = row['model_architecture']
            poisoned = 'backdoored' if bool(row['poisoned']) else 'clean'
            synth_labeling_params = dict(model_img_size=int(row['cnn_img_size_pixels']), temperature=3.0)

            for arch_prefix, sdn_type in dict_arch_type.items():
                if model_architecture.startswith(arch_prefix):
                    root = os.path.join(root_path, model_name)
                    model_path = os.path.join(root, 'model.pt')
                    data_path = os.path.join(root, 'clean_example_data')
                    Logger.log(f'Training {model_architecture}-sdn ({poisoned}) in {root}')

                    time_start = datetime.now()

                    dataset, sdn_type, model = read_model_directory(model_path, data_path, batch_size, test_ratio, device)

                    print('Labeling synthetic dataset...')
                    clean_images, clean_labels = synthetic_module.return_model_data_and_labels(model, synth_labeling_params, synthetic_data['clean'])
                    clean_data = sdaf.ManualData(sdaf.convert_to_pytorch_format(clean_images), clean_labels['soft'])

                    # trick: replace original train loader with the synthetic loader
                    synthetic_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=True, num_workers=dataset.num_workers)
                    dataset.train_loader = synthetic_loader
                    dataset.test_loader = synthetic_loader

                    train_trojai_sdn(dataset, model, root, device)
                    # train_trojai_sdn_with_svm(dataset, model, root, device, log=True)

                    time_end = datetime.now()
                    Logger.log(f'elapsed {time_end - time_start}\n')
    Logger.log('script ended')
    Logger.close()


if __name__ == "__main__":
    main()
