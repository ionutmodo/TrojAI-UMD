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

if socket.gethostname() != 'windows10':
    os.chdir('/umd')
sys.path.extend(['/umd', '/umd/architectures/', '/umd/tools/', '/umd/trojai/', '/umd/trojai/trojai'])
import warnings

warnings.filterwarnings("ignore")
# cwd = os.getcwd()
# sys.path.extend([cwd, os.path.join(cwd, 'architectures'), os.path.join(cwd, 'tools'), os.path.join(cwd, 'trojai'), os.path.join(cwd, 'trojai', 'trojai')])

from tools.logistics import *
from tools.data import create_backdoored_dataset
from architectures.LightSDN import LightSDN
from tools.network_architectures import load_trojai_model, save_model
import tools.umd_pipeline_tools as pipeline_tools
import tools.model_funcs as mf
import tools.aux_funcs as af
from datetime import datetime
import numpy as np
import argparse
from tools.logger import Logger
from concurrent.futures import ProcessPoolExecutor as Pool


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


def build_confusion_distribution_stats(scratch_dirpath, examples_dirpath, sdn_light, batch_size, device, perform_fast_test):
    if perform_fast_test:
        mean_clean, std_clean = np.random.uniform(low=0.0, high=1.0, size=2)
    else:
        dataset_clean = TrojAI(folder=examples_dirpath, test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
        confusion_clean = mf.compute_confusion(sdn_light, dataset_clean.train_loader, device)
        mean_clean, std_clean = np.mean(confusion_clean), np.std(confusion_clean)
        del dataset_clean, confusion_clean

    stats = {'mean_clean': mean_clean, 'std_clean': std_clean}
    for key in ['polygon', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']:
        path = os.path.join(scratch_dirpath, f'backdoored_data_{key}')
        if perform_fast_test:
            mean, mean_diff, std, std_diff = np.random.uniform(low=0.0, high=1.0, size=4)
        else:
            dataset = TrojAI(folder=path, test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
            confusion = mf.compute_confusion(sdn_light, dataset.train_loader, device)
            mean, std = np.mean(confusion), np.std(confusion)
            mean_diff, std_diff = abs(mean - mean_clean), abs(std - std_clean)
            del dataset, confusion
        stats[f'mean_{key}'], stats[f'mean_diff_{key}'] = mean, mean_diff
        stats[f'std_{key}'], stats[f'std_diff_{key}'] = std, std_diff
    return stats


def build_confusion_matrix_stats():
    stats = {}
    return stats


def trojan_detector_umd(model_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    STATISTIC_TYPE_RAW_MEAN_STD, STATISTIC_TYPE_DIFF_MEAN_STD, STATISTIC_TYPE_H_KL = 10, 11, 12
    SDN_IC_TYPE_SVM, SDN_IC_TYPE_FULLY_CONNECTED = 20, 21
    now = datetime.now
    time_start = now()

    print_messages = True
    sdn_ic_type = SDN_IC_TYPE_FULLY_CONNECTED
    add_arch_features = False
    # ADD ONE-HOT/RAW ARCH FEATURE
    fast_local_test = False
    stats_type = STATISTIC_TYPE_DIFF_MEAN_STD
    trigger_size = 30
    trigger_color = 'random' # 'random' or (127, 127, 127)
    path_meta_model = 'metamodels/metamodel_14_fc_round3_data=diffs_square=30-random_scaler=no_clf=rf-500_arch-features=no_exclude-sts=yes'

    batch_size, batch_size_sdn_training = 1, 1 # to avoid some warnings in PyCharm
    if sdn_ic_type == SDN_IC_TYPE_SVM:
        batch_size_sdn_training = 1
        batch_size = 1
    elif sdn_ic_type == SDN_IC_TYPE_FULLY_CONNECTED:
        hostname = socket.gethostname()
        batch_size_sdn_training = 10 if hostname == 'windows10' else 20
        batch_size = 1 if hostname == 'windows10' else 50
    device = af.get_pytorch_device()
    sdn_name = f'ics_train100_test0_bs{batch_size_sdn_training}'

    ################################################################################
    #################### STEP 1: train SDN
    ################################################################################
    if print_messages:
        print()
        # print(f'[info] reading clean dataset & model')
        print(f'[info] current folder is {os.getcwd()}')
        print(f'[info] model_filepath is {model_filepath}')
        print(f'[info] result_filepath is {result_filepath}')
        print(f'[info] scratch_dirpath is {scratch_dirpath}')
        print(f'[info] examples_dirpath is {examples_dirpath}')

    t = now()

    dataset_clean, sdn_type, model = read_model_directory(model_filepath, examples_dirpath, batch_size=batch_size_sdn_training, test_ratio=0, device=device)
    num_classes = dataset_clean.num_classes

    if print_messages:
        print(f'[info] reading clean dataset and raw model took {now() - t}')
        print()
        print('[info] STEP 1: train SDN with SVM ICs')

    t = now()
    if not fast_local_test:
        if sdn_ic_type == SDN_IC_TYPE_SVM:
            pipeline_tools.train_trojai_sdn_with_svm(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=device, log=print_messages)
        elif sdn_ic_type == SDN_IC_TYPE_FULLY_CONNECTED:
            pipeline_tools.train_trojai_sdn_with_fc(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=device)
    del dataset_clean
    if print_messages:
        print(f'[info] training SDN took {now() - t}')

    ################################################################################
    #################### STEP 2: create backdoored datasets
    ################################################################################
    ### the speed can be improved by creating the datasets using multiprocessing (1 process for each dataset to be created)
    # create polygon dataset and save it to disk
    if print_messages:
        print()
        print('[info] STEP 2: create backdoored datasets')

    t = now()
    if not fast_local_test:
        build_datasets(examples_dirpath, scratch_dirpath, trigger_size, trigger_color, 0)

    if print_messages:
        print(f'[info] creating all backdoored datasets took {now() - t}')

    ################################################################################
    #################### STEP 3: create backdoored datasets
    ################################################################################
    # create dataset paths

    if print_messages:
        print()
        print('[info] STEP 3: loading backdoored datasets, computing confusion distribution')

    t = now()
    model = None
    if stats_type in [STATISTIC_TYPE_RAW_MEAN_STD, STATISTIC_TYPE_DIFF_MEAN_STD]:
        if sdn_ic_type == SDN_IC_TYPE_SVM:
            model = LightSDN(path_model_cnn=model_filepath, path_model_ics=os.path.join(scratch_dirpath, 'ics_svm.model'), sdn_type=sdn_type, num_classes=num_classes, device=device)
        elif sdn_ic_type == SDN_IC_TYPE_FULLY_CONNECTED:
            model = load_trojai_model(sdn_path=os.path.join(scratch_dirpath, sdn_name), cnn_path=model_filepath, num_classes=num_classes, sdn_type=sdn_type, device=device)
    elif stats_type == STATISTIC_TYPE_H_KL:
        model = torch.load(model_filepath, map_location=device).eval()

    if print_messages:
        print(f'[info] loading light SDN took {now() - t}')

    t = now()
    stats = None
    if stats_type in [STATISTIC_TYPE_RAW_MEAN_STD, STATISTIC_TYPE_DIFF_MEAN_STD]:
        stats = build_confusion_distribution_stats(scratch_dirpath, examples_dirpath, model, batch_size, device, fast_local_test)
    elif stats_type == STATISTIC_TYPE_H_KL:
        stats = build_confusion_matrix_stats()

    if print_messages:
        print()
        print(f'[info] computing confusion distribution for clean, polygon and filters took {now() - t}')

    ## DIFF FEATURES
    features_diff = np.array([
        stats['mean_diff_polygon'], stats['std_diff_polygon'],
        stats['mean_diff_gotham'], stats['std_diff_gotham'],
        stats['mean_diff_kelvin'], stats['std_diff_kelvin'],
        stats['mean_diff_lomo'], stats['std_diff_lomo'],
        stats['mean_diff_nashville'], stats['std_diff_nashville'],
        stats['mean_diff_toaster'], stats['std_diff_toaster'],
    ]).reshape(1, -1)

    ## RAW FEATURES
    features_raw = np.array([
        stats['mean_clean'],     stats['std_clean'],
        stats['mean_polygon'],   stats['std_polygon'],
        stats['mean_gotham'],    stats['std_gotham'],
        stats['mean_kelvin'],    stats['std_kelvin'],
        stats['mean_lomo'],      stats['std_lomo'],
        stats['mean_nashville'], stats['std_nashville'],
        stats['mean_toaster'],   stats['std_toaster'],
    ]).reshape(1, -1)

    ### SETTING FEATURES VARIABLE
    if stats_type == STATISTIC_TYPE_DIFF_MEAN_STD:
        features = features_diff
    elif stats_type == STATISTIC_TYPE_RAW_MEAN_STD:
        features = features_raw
    elif stats_type == STATISTIC_TYPE_H_KL:
        pass
        # features = features_h_kl

    if print_messages:
        print(f'[info] computed features for model {model_filepath.split(os.path.sep)[-2]}')
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

    ################################################################################
    #################### STEP 4: predict backdoor probability
    ################################################################################
    if print_messages:
        print()
        print('[info] STEP 4: predicting backd proba')

    # check if scaler exists
    scaler = af.load_obj(os.path.join(path_meta_model, 'scaler.pickle'))
    if scaler is not None:
        features = scaler.transform(features)
        print('all features after scaling:', features.tolist())

    if add_arch_features:
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
        arch_one_hot = np.identity(len(available_architectures)).tolist()[arch_code]
        features = features[0].tolist()  # do this because features has size (1, N)
        features = np.array(arch_one_hot + features).reshape(1, -1)
        print(f'[one-hot] arch: {available_architectures[sdn_type]}, one-hot: {arch_one_hot}')
        print(f'[feature] final features: {features.tolist()}')

    meta_model = af.load_obj(filename=os.path.join(path_meta_model, 'model.pickle'))
    positive_class_index = np.where(meta_model.classes_ == 1)[0][0]  # only for sklearn models
    probabilities = meta_model.predict_proba(features)
    backd_proba = probabilities[0][positive_class_index]

    if print_messages:
        print(f'[info] probability distribution: {probabilities}')
        print(f'[info] predicted backdoor probability: {backd_proba}')

    ### write prediction to file
    write_prediction(result_filepath, str(backd_proba))
    time_end = now()
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
