"""
        This file contains the processing pipeline to get the backdoor probability for a model.
        Our approach uses the SDN modification of an off the shelf CNN to compute an empirical
    discrete distribution called "confusion distribution". For a given dataset, for every single
    image in the dataset we compute a confusion score which quantifies the internal disagreements
    between internal classifiers and the original output of the CNN.
        Our idea states that backdoored models will tend to increase the confusion score for the
    backdoored images or a similar pattern. Of course, during test time we don't know what the
    trigger is, but we know its type, which is polygon or instagram filter. In order to approach
    the real scenario, we are simulating (or approximating) the polygon trigger with a square
    placed in the middle. The square color is black for clean data and has the original color the
    model was trained with for the backdoored models.
        Our pipeline:
    1. train the SDN
    2. create the backdoored datasets for each trigger: square-size-N and 5 instagram filters
    3. a) compute the confusion distribution for all datasets (clean, polygon, filters)
       b) compute features for the meta-classifier
    4. use the features from step 3 to get a backdoor probability using the meta-classifier
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow messages
import sys
import socket

root = None
if socket.gethostname() != 'windows10':
    if os.path.isdir('/umd'):
        root = '/umd'
    if os.path.isdir('/TrojAI-UMD'):
        root = '/TrojAI-UMD'
if root is not None:
    os.chdir(root)
    sys.path.extend([root,
                     os.path.join(root, 'architectures'),
                     os.path.join(root, 'tools'),
                     os.path.join(root, 'trojai'),
                     os.path.join(root, 'trojai/trojai')])
print(sys.path)
import warnings

warnings.filterwarnings("ignore")
# cwd = os.getcwd()
# sys.path.extend([cwd, os.path.join(cwd, 'architectures'), os.path.join(cwd, 'tools'), os.path.join(cwd, 'trojai'), os.path.join(cwd, 'trojai', 'trojai')])

from tools.logistics import *
from tools.data import create_backdoored_dataset
from architectures.LightSDN import LightSDN
from tools.network_architectures import load_trojai_model
import tools.umd_pipeline_tools as pipeline_tools
import tools.model_funcs as mf
import tools.aux_funcs as af
from datetime import datetime
import numpy as np
import argparse
from scipy.stats import entropy
import synthetic_data.gen_backdoored_datasets as synthetic_module
import synthetic_data.aux_funcs as sdaf
from notebooks.methods import keras_load
# from concurrent.futures import ProcessPoolExecutor as Pool


def write_prediction(filepath, backd_proba):
    with open(filepath, 'w') as w:
        w.write(backd_proba)


def build_datasets(examples_dirpath, scratch_dirpath, trigger_size, trigger_color, trigger_target_class):
    create_backdoored_dataset(dir_clean_data=examples_dirpath,
                              dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_polygon'),
                              trigger_type='polygon',
                              trigger_name='square',
                              trigger_color=trigger_color,
                              trigger_size=trigger_size,
                              triggered_classes='all',
                              trigger_target_class=trigger_target_class)

    # create filters dataset and save it to disk
    for p_filter in ['gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
        create_backdoored_dataset(dir_clean_data=examples_dirpath,
                                  dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_{p_filter}'),
                                  trigger_type='filter',
                                  trigger_name=p_filter,
                                  trigger_color=None,
                                  trigger_size=None,
                                  triggered_classes='all',
                                  trigger_target_class=trigger_target_class)


def build_confusion_distribution_stats(scratch_dirpath, examples_dirpath, model, batch_size, device, perform_fast_test):
    stats = {}
    for key in ['clean', 'polygon_all', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
        path = examples_dirpath if key == 'clean' else os.path.join(scratch_dirpath, f'backdoored_data_{key}')
        if perform_fast_test:
            mean, mean_diff, std, std_diff = np.random.uniform(low=0.0, high=1.0, size=4)
        else:
            dataset = TrojAI(folder=path, test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
            confusion = mf.compute_confusion(model, dataset.train_loader, device)
            mean, std = np.mean(confusion), np.std(confusion)
            if key != 'clean':
                mean_diff, std_diff = abs(mean - stats['mean_clean']), abs(std - stats['std_clean'])
                stats[f'mean_diff_{key}'], stats[f'std_diff_{key}'] = mean_diff, std_diff
            del dataset, confusion
        stats[f'mean_{key}'], stats[f'std_{key}'] = mean, std
    return stats


def build_confusion_distribution_stats_synthetic(synthetic_data, model, batch_size, device, perform_fast_test, use_abs=False):
    def f(x): return abs(x) if use_abs else x
    stats = {}
    for key_synth_data in ['clean', 'polygon_all', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
        key = key_synth_data.replace('_all', '')
        if perform_fast_test:
            mean, mean_diff, std, std_diff = np.random.uniform(low=0.0, high=1.0, size=4)
        else:
            n_samples = synthetic_data[key_synth_data].shape[0]
            # computing conf. distribution doesn't require labels (use some fake labels here: zeros)
            data = sdaf.ManualData(sdaf.convert_to_pytorch_format(synthetic_data[key_synth_data]), np.zeros((n_samples, )))
            synthetic_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

            confusion = mf.compute_confusion(model, synthetic_loader, device)
            mean, std = np.mean(confusion), np.std(confusion)
            del confusion
        if key != 'clean':
            mean_diff, std_diff = f(mean - stats['mean_clean']), f(std - stats['std_clean'])
            stats[f'mean_diff_{key}'], stats[f'std_diff_{key}'] = mean_diff, std_diff
        stats[f'mean_{key}'], stats[f'std_{key}'] = mean, std
    return stats


def get_confusion_matrix_stats(model, dataset, device):
    nc = dataset.num_classes
    matrix = np.zeros((nc, nc), dtype=np.int64)

    for image, label_true in dataset.train_loader:
        outputs = model(image.to(device))
        for i, out in enumerate(outputs):
            label_pred = out.unsqueeze(0).max(1)[1].item()
            matrix[label_true[i].item(), label_pred] += 1
        del outputs
        torch.cuda.empty_cache()

    column_mean = matrix.mean(axis=0)
    proba = column_mean / column_mean.sum()

    uniform = np.ones_like(proba) / nc
    h = entropy(proba)
    kl = entropy(proba, uniform)

    return h / nc, kl / nc


def build_confusion_matrix_stats(scratch_dirpath, examples_dirpath, model, batch_size, device, perform_fast_test):
    stats = {}
    for key in ['clean', 'polygon', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
        path = examples_dirpath if key == 'clean' else os.path.join(scratch_dirpath, f'backdoored_data_{key}')
        if perform_fast_test:
            h, kl = np.random.uniform(low=0.0, high=1.0, size=2)
        else:
            dataset = TrojAI(folder=path, test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
            h, kl = get_confusion_matrix_stats(model, dataset, device)
            del dataset
            torch.cuda.empty_cache()
        stats[f'h_{key}'], stats[f'kl_{key}'] = h, kl
    return stats


def get_features_from_stats(stats, network_type, stats_type, print_messages):
    features = None
    if network_type in [NETWORK_TYPE_SDN_WITH_SVM_ICS, NETWORK_TYPE_SDN_WITH_FC_ICS]:
        features_diff = np.array([
            stats['mean_diff_polygon'],   stats['std_diff_polygon'],
            stats['mean_diff_gotham'],    stats['std_diff_gotham'],
            stats['mean_diff_kelvin'],    stats['std_diff_kelvin'],
            stats['mean_diff_lomo'],      stats['std_diff_lomo'],
            stats['mean_diff_nashville'], stats['std_diff_nashville'],
            stats['mean_diff_toaster'],   stats['std_diff_toaster']
        ]).reshape(1, -1)

        features_raw = np.array([
            stats['mean_clean'],     stats['std_clean'],
            stats['mean_polygon'],   stats['std_polygon'],
            stats['mean_gotham'],    stats['std_gotham'],
            stats['mean_kelvin'],    stats['std_kelvin'],
            stats['mean_lomo'],      stats['std_lomo'],
            stats['mean_nashville'], stats['std_nashville'],
            stats['mean_toaster'],   stats['std_toaster']
        ]).reshape(1, -1)

        if stats_type == STATISTIC_TYPE_RAW_MEAN_STD:
            features = features_raw
        elif stats_type == STATISTIC_TYPE_DIFF_MEAN_STD:
            features = features_diff

        if print_messages:
            print('[raw  feature] clean:',     stats['mean_clean'],     stats['std_clean'])
            print('[raw  feature] polygon:',   stats['mean_polygon'],   stats['std_polygon'])
            print('[raw  feature] gotham:',    stats['mean_gotham'],    stats['std_gotham'])
            print('[raw  feature] kelvin:',    stats['mean_kelvin'],    stats['std_kelvin'])
            print('[raw  feature] lomo:',      stats['mean_lomo'],      stats['std_lomo'])
            print('[raw  feature] nashville:', stats['mean_nashville'], stats['std_nashville'])
            print('[raw  feature] toaster:',   stats['mean_toaster'],   stats['std_toaster'])
            print('[raw  feature] raw features:', features_raw.tolist())
            print()
            print('[diff feature] polygon:',   stats['mean_diff_polygon'],   stats['std_diff_polygon'])
            print('[diff feature] gotham:',    stats['mean_diff_gotham'],    stats['std_diff_gotham'])
            print('[diff feature] kelvin:',    stats['mean_diff_kelvin'],    stats['std_diff_kelvin'])
            print('[diff feature] lomo:',      stats['mean_diff_lomo'],      stats['std_diff_lomo'])
            print('[diff feature] nashville:', stats['mean_diff_nashville'], stats['std_diff_nashville'])
            print('[diff feature] toaster:',   stats['mean_diff_toaster'],   stats['std_diff_toaster'])
            print('[diff feature] diff features:', features_diff.tolist())
    elif network_type == NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING:
        features_h = np.array([
            stats['h_clean'],
            stats['h_polygon'],
            stats['h_gotham'],
            stats['h_kelvin'],
            stats['h_lomo'],
            stats['h_nashville'],
            stats['h_toaster']
        ]).reshape(1, -1)

        features_kl = np.array([
            stats['kl_clean'],
            stats['kl_polygon'],
            stats['kl_gotham'],
            stats['kl_kelvin'],
            stats['kl_lomo'],
            stats['kl_nashville'],
            stats['kl_toaster'],
        ]).reshape(1, -1)

        features_h_kl = np.array([
            stats['h_clean'],     stats['kl_clean'],
            stats['h_polygon'],   stats['kl_polygon'],
            stats['h_gotham'],    stats['kl_gotham'],
            stats['h_kelvin'],    stats['kl_kelvin'],
            stats['h_lomo'],      stats['kl_lomo'],
            stats['h_nashville'], stats['kl_nashville'],
            stats['h_toaster'],   stats['kl_toaster'],
        ]).reshape(1, -1)

        if stats_type == STATISTIC_TYPE_H:
            features = features_h
        elif stats_type == STATISTIC_TYPE_KL:
            features = features_kl
        elif stats_type == STATISTIC_TYPE_H_KL:
            features = features_h_kl

        if print_messages:
            print('[H/KL  features] clean:',     stats['h_clean'],     stats['kl_clean'])
            print('[H/KL  features] polygon:',   stats['h_polygon'],   stats['kl_polygon'])
            print('[H/KL  features] gotham:',    stats['h_gotham'],    stats['kl_gotham'])
            print('[H/KL  features] kelvin:',    stats['h_kelvin'],    stats['kl_kelvin'])
            print('[H/KL  features] lomo:',      stats['h_lomo'],      stats['kl_lomo'])
            print('[H/KL  features] nashville:', stats['h_nashville'], stats['kl_nashville'])
            print('[H/KL  features] toaster:',   stats['h_toaster'],   stats['kl_toaster'])
    return features


def trojan_detector_umd(model_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    time_start = datetime.now()
    SCENARIOS = {
        1: (NETWORK_TYPE_SDN_WITH_FC_ICS, STATISTIC_TYPE_DIFF_MEAN_STD),
        2: (NETWORK_TYPE_SDN_WITH_FC_ICS, STATISTIC_TYPE_RAW_MEAN_STD),

        3: (NETWORK_TYPE_SDN_WITH_SVM_ICS, STATISTIC_TYPE_DIFF_MEAN_STD),
        4: (NETWORK_TYPE_SDN_WITH_SVM_ICS, STATISTIC_TYPE_RAW_MEAN_STD),

        5: (NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING, STATISTIC_TYPE_H),
        6: (NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING, STATISTIC_TYPE_KL),
        7: (NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING, STATISTIC_TYPE_H_KL),
    }

    synthetic_data = np.load('synthetic_data/synthetic_data_1000_clean_polygon_instagram.npz')
    ################################################################################
    #################### EXPERIMENT SETTINGS
    ################################################################################
    print_messages = True
    fast_local_test = False
    arch_wise_metamodel = False # used to specify if we have one metamodel per architecture
    use_abs_features = False # compute abs for diff features

    add_arch_features = True # ALSO ADD ONE-HOT/RAW ARCH FEATURE
    scenario_number = 1
    trigger_size = 30
    trigger_color = 'random' # 'random' or (127, 127, 127)
    path_meta_model = 'metamodels/metamodel_18-2_fc_round4_data=synth-diffs_scaler=no_clf=NN_arch-features=yes_arch-wise-models=no_out=bernoulli'

    model_output_type = None
    if 'out=binary' in path_meta_model:
        model_output_type = 'binary'
    elif 'out=bernoulli' in path_meta_model:
        model_output_type = 'bernoulli'
    elif 'out=2x-bernoulli' in path_meta_model:
        model_output_type = '2x-bernoulli'
    elif 'out=2x-softmax' in path_meta_model:
        model_output_type = '2x-softmax'

    network_type, stats_type = SCENARIOS[scenario_number]
    batch_size_training, batch_size_experiment = 1, 1 # to avoid some warnings in PyCharm
    if network_type == NETWORK_TYPE_SDN_WITH_SVM_ICS:
        batch_size_training, batch_size_experiment = 1, 1
    elif network_type == NETWORK_TYPE_SDN_WITH_FC_ICS:
        batch_size_training, batch_size_experiment = (10, 5) if socket.gethostname() == 'windows10' else (20, 50)
    elif network_type == NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING:
        batch_size_training, batch_size_experiment = (1, 1) if socket.gethostname() == 'windows10' else (20, 50) # batch_size_training is not used

    device = af.get_pytorch_device()
    # sdn_name = f'ics_train100_test0_bs{batch_size_training}' # only used in scenarios 1, 2, 3, 4
    sdn_name = 'ics_synthetic-1000_train100_test0_bs20'

    # # speedup for ES vs STS: print messages only for STS (models #0 and #1) and disable printing messages for other models (for ES)
    current_model_name = model_filepath.split(os.path.sep)[-2]
    if current_model_name not in ['id-00000000', 'id-00000001']:
        print_messages = False

    if print_messages:
        print(f'[info] current folder is {os.getcwd()}')
        print(f'[info] model_filepath is {model_filepath}')
        print(f'[info] result_filepath is {result_filepath}')
        print(f'[info] scratch_dirpath is {scratch_dirpath}')
        print(f'[info] examples_dirpath is {examples_dirpath}')
        print()

    ##########################################################################################
    #################### STEP 1: train SDN
    ##########################################################################################
    dataset_clean, sdn_type, model = read_model_directory(model_filepath, examples_dirpath, batch_size=batch_size_training, test_ratio=0, device=device)
    # label synthetic dataset

    if not fast_local_test:
        synth_labeling_params = dict(model_img_size=244, temperature=3.0)
        clean_images, clean_labels = synthetic_module.return_model_data_and_labels(model, synth_labeling_params, synthetic_data['clean'])
        clean_data = sdaf.ManualData(sdaf.convert_to_pytorch_format(clean_images), clean_labels['soft'])

        # trick: replace original train loader with the synthetic loader
        synthetic_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size_training, shuffle=True, num_workers=dataset_clean.num_workers)
        dataset_clean.train_loader = synthetic_loader
        dataset_clean.test_loader = synthetic_loader

    num_classes = dataset_clean.num_classes # will be used when we load the SDN model
    if network_type == NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING:
        # don't perform any training because we use the raw neural network that we are provided in the model.pt file
        print('[info] STEP 1: using raw CNN that we are provided')
    elif network_type == NETWORK_TYPE_SDN_WITH_SVM_ICS:
        print('[info] STEP 1: train SDN with SVM ICs')
        if not fast_local_test:
            pipeline_tools.train_trojai_sdn_with_svm(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=device, log=print_messages)
    elif network_type == NETWORK_TYPE_SDN_WITH_FC_ICS:
        print('[info] STEP 1: train SDN with FC ICs')
        if not fast_local_test:
            pipeline_tools.train_trojai_sdn_with_fc(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=device)
    del dataset_clean

    ##########################################################################################
    #################### STEP 2: create backdoored datasets
    ##########################################################################################
    ### the speed can be improved by creating the datasets using multiprocessing (1 process for each dataset to be created)
    if print_messages:
        print('\n[info] STEP 2: create backdoored datasets')

    ## uncomment this when you use the provided data (not synthetic dataset)
    # if not fast_local_test:
    #     build_datasets(examples_dirpath, scratch_dirpath, trigger_size, trigger_color, 0)

    ##########################################################################################
    #################### STEP 3: load neural network and compute feature statistics
    ##########################################################################################
    if print_messages:
        print(f'\n[info] STEP 3: computing confusion distribution for model {current_model_name}')

    if network_type == NETWORK_TYPE_SDN_WITH_SVM_ICS:
        model = LightSDN(path_model_cnn=model_filepath, path_model_ics=os.path.join(scratch_dirpath, 'ics_svm.model'), sdn_type=sdn_type, num_classes=num_classes, device=device)
    elif network_type == NETWORK_TYPE_SDN_WITH_FC_ICS:
        model = load_trojai_model(sdn_path=os.path.join(scratch_dirpath, sdn_name), cnn_path=model_filepath, num_classes=num_classes, sdn_type=sdn_type, device=device)
    elif network_type == NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING:
        model = torch.load(model_filepath, map_location=device).eval()
    else:
        raise RuntimeError('Invalid value for variable "network_type"')

    if (network_type in [NETWORK_TYPE_SDN_WITH_SVM_ICS, NETWORK_TYPE_SDN_WITH_FC_ICS]) and (stats_type in [STATISTIC_TYPE_RAW_MEAN_STD, STATISTIC_TYPE_DIFF_MEAN_STD]):
        stats = build_confusion_distribution_stats_synthetic(synthetic_data, model, batch_size_experiment, device, fast_local_test, use_abs=use_abs_features)
    elif (network_type == NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING) and (stats_type in [STATISTIC_TYPE_H, STATISTIC_TYPE_KL, STATISTIC_TYPE_H_KL]):
        stats = build_confusion_matrix_stats(scratch_dirpath, examples_dirpath, model, batch_size_experiment, device, fast_local_test)
    else:
        raise RuntimeError('Invalid combination for variables "network_type" and "stats_type"')

    features = get_features_from_stats(stats, network_type, stats_type, print_messages)

    ##########################################################################################
    #################### STEP 4: predict backdoor probability
    ##########################################################################################

    # do not change the values in this dictionary!
    available_architectures = {
        SDNConfig.DenseNet_blocks: 'densenet',
        SDNConfig.GoogLeNet: 'googlenet',
        SDNConfig.Inception3: 'inception',
        SDNConfig.MobileNet2: 'mobilenet',
        SDNConfig.ResNet: 'resnet',
        SDNConfig.ShuffleNet: 'shufflenet',
        SDNConfig.SqueezeNet: 'squeezenet',
        SDNConfig.VGG: 'vgg'
    }
    arch_code = pipeline_tools.encode_architecture(available_architectures[sdn_type])

    if print_messages:
        print('\n[info] STEP 4: predicting backd proba')
        print(f'[info] model code is {arch_code} ({available_architectures[arch_code]})')

    suffix = '' # if the meta model is trained on all model architectures, the files will be model.pickle and scaler.pickle. Otherwise, append this suffix
    if arch_wise_metamodel:
        suffix = f'-{available_architectures[arch_code]}' # let that dash there, such that the result would be, for example, model-vgg.pickle and scaler-vgg.pickle

    # meta_model = af.load_obj(filename=os.path.join(path_meta_model, f'model{suffix}.pickle'))
    meta_model = keras_load(path_meta_model)
    scaler = af.load_obj(os.path.join(path_meta_model, f'scaler{suffix}.pickle'))

    if scaler is not None:
        features = scaler.transform(features)
        print('[feature] scaled features:', features.tolist())

    if add_arch_features:
        arch_one_hot = np.identity(len(available_architectures)).tolist()[arch_code]
        features = features[0].tolist()  # do this because features has size (1, N)
        features = np.array(features + arch_one_hot).reshape(1, -1)
        print(f'[one-hot] arch: {available_architectures[sdn_type]}, one-hot: {arch_one_hot}')
        print(f'[feature] final features: {features.tolist()}')

    ## KERAS MODEL
    if model_output_type == 'binary':
        backd_proba = meta_model.predict(features)[0][0]
    elif model_output_type == 'bernoulli':
        prediction = meta_model.predict(features)[0]
        pair_label_prediction = sorted(enumerate(prediction), key=lambda x: -x[1])
        label, proba = pair_label_prediction[0]
        if label == 0: # clean has max probability => predict 1 - proba
            backd_proba = 1.0 - proba
        else: # a backdoored class has max probability => predict proba
            backd_proba = proba
    elif model_output_type == '2x-bernoulli':
        pass # not yet implemented
    elif model_output_type == '2x-softmax':
        pass # not yet implemented

    ## SKLEARN MODEL
    # positive_class_index = np.where(meta_model.classes_ == 1)[0][0]
    # probabilities = meta_model.predict_proba(features)
    # backd_proba = probabilities[0][positive_class_index]

    if print_messages:
        print(f'[info] model code is {arch_code}')
        # print(f'[info] probability distribution: {probabilities}')
        print(f'[info] predicted backdoor probability: {backd_proba}')

    ### write prediction to file
    write_prediction(result_filepath, str(backd_proba)) # try 1-backd_proba
    time_end = datetime.now()
    if print_messages:
        print(f'[info] script ended (elapsed {time_end - time_start})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, default='./model.pt', help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath', type=str, default='./output',
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, default='./scratch',
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, default='./example', help='File path to the folder of examples which might be useful for determining whether a model is poisoned.')

    args = parser.parse_args()

    STATISTIC_TYPE_RAW_MEAN_STD, STATISTIC_TYPE_DIFF_MEAN_STD, STATISTIC_TYPE_H_KL, STATISTIC_TYPE_H, STATISTIC_TYPE_KL = 10, 11, 12, 13, 14
    NETWORK_TYPE_SDN_WITH_SVM_ICS, NETWORK_TYPE_SDN_WITH_FC_ICS, NETWORK_TYPE_RAW_CNN_NO_ADDITIONAL_TRAINING = 20, 21, 22

    trojan_detector_umd(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)

# TODO: set a limit for the number of images per class when reading them from disk (Sanghyun's idea with 1,3,5 images per class)
# sudo singularity build umd_pipeline.simg umd_pipeline.def
# sudo singularity run -B /home/ubuntu/workplace/TrojAI-data/id-00001000/ /home/ubuntu/workplace/TrojAI-UMD/06_round3_rbf-svm_size30_RANDOM_all-classes.simg --model_filepath /home/ubuntu/workplace/TrojAI-data/id-00001000/model.pt --result_filepath /home/ubuntu/workplace/TrojAI-data/id-00001000/result.txt --scratch_dirpath /home/ubuntu/workplace/TrojAI-data/id-00001000/scratch --examples_dirpath /home/ubuntu/workplace/TrojAI-data/id-00001000/clean_example_data

# mp_mapping_params = [dict(
#     dir_clean_data=examples_dirpath,
#     dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_square_{trigger_size}'),
#     trigger_type='polygon',
#     trigger_name='square',
#     trigger_color=trigger_color,
#     trigger_size=trigger_size,
#     triggered_classes='all',
#     trigger_target_class=trigger_target_class)]
#
# # create filters dataset and save it to disk
# for p_filter in list_filters:
#     mp_mapping_params.append(dict(
#         dir_clean_data=examples_dirpath,
#         dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_filter_{p_filter}'),
#         trigger_type='filter',
#         trigger_name=p_filter,
#         trigger_color=None,
#         trigger_size=None,
#         triggered_classes='all',
#         trigger_target_class=trigger_target_class)
#     )
#
# with mp.Pool(processes=len(mp_mapping_params)) as pool:
#     pool.map(worker_create_dataset, mp_mapping_params)

# --model_filepath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000000\model.pt" --result_filepath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000000\scratch\result.txt" --scratch_dirpath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000000\scratch" --examples_dirpath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000000\clean_example_data
# --model_filepath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000001\model.pt" --result_filepath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000001\scratch\result.txt" --scratch_dirpath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000001\scratch" --examples_dirpath "D:\Cloud\MEGA\TrojAI\TrojAI-data\round3-train-dataset\id-00000001\clean_example_data

# def worker_backdoored_dataset_creator(params):
#     create_backdoored_dataset(**params)
#
#
# def parallelize_backdoored_dataset_creation(p_examples_dirpath, p_scratch_dirpath, p_trigger_size, p_trigger_color, p_trigger_target_class, p_list_filters):
#     print('[info] parallelizing...')
#     mp_mapping_params = [dict(dir_clean_data=p_examples_dirpath,
#                               dir_backdoored_data=os.path.join(p_scratch_dirpath, f'backdoored_data_square_{p_trigger_size}'),
#                               trigger_type='polygon',
#                               trigger_name='square',
#                               trigger_color=p_trigger_color,
#                               trigger_size=p_trigger_size,
#                               triggered_classes='all',
#                               trigger_target_class=p_trigger_target_class)]
#
#     # create filters dataset and save it to disk
#     for p_filter in p_list_filters:
#         mp_mapping_params.append(dict(dir_clean_data=p_examples_dirpath,
#                                       dir_backdoored_data=os.path.join(p_scratch_dirpath, f'backdoored_data_filter_{p_filter}'),
#                                       trigger_type='filter',
#                                       trigger_name=p_filter,
#                                       trigger_color=None,
#                                       trigger_size=None,
#                                       triggered_classes='all',
#                                       trigger_target_class=p_trigger_target_class))
#
#     with Pool(max_workers=len(mp_mapping_params)) as pool:
#         pool.map(worker_backdoored_dataset_creator, mp_mapping_params)
