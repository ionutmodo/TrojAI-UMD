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
       b) compute mean_diff and std_diff statistics between each confusion distribution and clean one
    4. use the values from step 5 to get a prediction using the binary meta-classifier
"""
import os
import sys
os.chdir('/umd')
sys.path.extend(['/umd', '/umd/architectures/', '/umd/tools/', '/umd/trojai/', '/umd/trojai/trojai'])
# cwd = os.getcwd()
# sys.path.extend([cwd, os.path.join(cwd, 'architectures'), os.path.join(cwd, 'tools'), os.path.join(cwd, 'trojai'), os.path.join(cwd, 'trojai', 'trojai')])

from tools.logistics import *
from tools.data import create_backdoored_dataset
from architectures.LightSDN import LightSDN
import tools.model_funcs as mf
import tools.aux_funcs as af
from datetime import datetime
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tools.logger import Logger


def now():
    return datetime.now()


def get_mean_std_diffs(confusion, clean_mean, clean_std, use_abs):
    conf_mean = np.mean(confusion)
    conf_std = np.std(confusion)

    diff_mean = conf_mean - clean_mean
    diff_std = conf_std - clean_std

    if use_abs:
        return abs(diff_mean), abs(diff_std)
    return diff_mean, diff_std


def write_prediction(filepath, backd_proba):
    with open(filepath, 'w') as w:
        w.write(backd_proba)


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

    print(f'[info] labels shape: {n_classes}')
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

            Logger.log(f'[info] SVM-IC-{i} Raw Acc: [{", ".join(list_raw_acc)}]')
            Logger.log(f'[info] SVM-IC-{i} Bal Acc: [{", ".join(list_bal_acc)}]')
            Logger.log(f'--------------------------------------------------------------------------')

    path_svm_model = os.path.join(model_root_path, 'ics_svm.model')
    af.save_obj(obj=svm_ics, filename=path_svm_model)
    size = os.path.getsize(path_svm_model) / (2 ** 20)
    if log:
        Logger.log(f'[info] SVM model ({size:.2f} MB) saved to {path_svm_model}')


def trojan_detector_umd(model_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    time_start = datetime.now()
    print_messages = True
    use_abs_for_diff_features = True
    trigger_size = 25 # for polygon dataset
    trigger_color = (0, 0, 0) # also try (127, 127, 127) or random (R, G, B) files
    trigger_target_class = 0 # can be anything, its used just for the new file name
    list_filters = ['gotham', 'kelvin', 'lomo', 'nashville', 'toaster']
    path_meta_model = 'metamodel_svm_square25_filters_black_square.pickle' # will be loaded from current directory, whatever it is
    batch_size = 1 # do not change this!
    _device = af.get_pytorch_device()

    ################################################################################
    #################### STEP 1: train SDN
    ################################################################################
    if print_messages:
        print()
        print(f'[info] reading clean dataset & model')
        print(f'[info] current folder is {os.getcwd()}')
        print(f'[info] model_filepath is {model_filepath}')
        print(f'[info] result_filepath is {result_filepath}')
        print(f'[info] scratch_dirpath is {scratch_dirpath}')
        print(f'[info] examples_dirpath is {examples_dirpath}')

    t = now()
    dataset_clean, sdn_type, model = read_model_directory(model_filepath, examples_dirpath, batch_size=batch_size, test_ratio=0, device=_device)
    if print_messages:
        print(f'[info] read_model_directory took {now() - t}')
    num_classes = dataset_clean.num_classes

    if print_messages:
        print('[info] training SDN with SVM ICs')
    # this method saves the SVM-ICs to "scratch_dirpath"/svm/svm_models (the file has no extension)
    t = now()
    train_trojai_sdn_with_svm(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=_device, log=print_messages)
    if print_messages:
        print(f'[info] train_trojai_sdn_with_svm took {now() - t}')

    ################################################################################
    #################### STEP 2: create backdoored datasets
    ################################################################################
    ### the speed can be improved by creating the datasets using multiprocessing (1 process for each dataset to be created)
    # create polygon dataset and save it to disk
    if print_messages:
        print()
        print('[info] creating polygon dataset')
    t = now()
    create_backdoored_dataset(dir_clean_data=examples_dirpath,
                              dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_polygon_{trigger_size}'),
                              trigger_type='polygon',
                              trigger_name='square',
                              trigger_color=trigger_color,
                              trigger_size=trigger_size,
                              triggered_classes='all',
                              trigger_target_class=trigger_target_class)
    if print_messages:
        print(f'[info] create_backdoored_dataset-polygon took {now() - t}')

    # create filters dataset and save it to disk
    for p_filter in list_filters:
        if print_messages:
            print('[info] creating dataset for filter', p_filter)
        t = now()
        create_backdoored_dataset(dir_clean_data=examples_dirpath,
                                  dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_filter_{p_filter}'),
                                  trigger_type='filter',
                                  trigger_name=p_filter,
                                  trigger_color=None,
                                  trigger_size=None,
                                  triggered_classes='all',
                                  trigger_target_class=trigger_target_class)
        if print_messages:
            print(f'[info] create_backdoored_dataset-filter {p_filter} took {now() - t}')

    ################################################################################
    #################### STEP 3: create backdoored datasets
    ################################################################################
    # create paths
    path_polygon   = os.path.join(scratch_dirpath, f'backdoored_data_polygon_{trigger_size}')
    path_gotham    = os.path.join(scratch_dirpath, f'backdoored_data_filter_gotham')
    path_kelvin    = os.path.join(scratch_dirpath, f'backdoored_data_filter_kelvin')
    path_lomo      = os.path.join(scratch_dirpath, f'backdoored_data_filter_lomo')
    path_nashville = os.path.join(scratch_dirpath, f'backdoored_data_filter_nashville')
    path_toaster   = os.path.join(scratch_dirpath, f'backdoored_data_filter_toaster')

    if print_messages:
        print()
        print('[info] STEP 3: loading backdoored datasets')
    # the clean dataset is loaded at the beginning
    t = now()
    dataset_polygon   = TrojAI(folder=path_polygon,   test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    dataset_gotham    = TrojAI(folder=path_gotham,    test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    dataset_kelvin    = TrojAI(folder=path_kelvin,    test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    dataset_lomo      = TrojAI(folder=path_lomo,      test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    dataset_nashville = TrojAI(folder=path_nashville, test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    dataset_toaster   = TrojAI(folder=path_toaster,   test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    if print_messages:
        print(f'[info] loading datasets polygon and filters took {now() - t}')

    # load model
    if print_messages:
        print('[info] loading SDN')
    t = now()
    path_model_cnn = model_filepath
    path_model_ics = os.path.join(scratch_dirpath, 'ics_svm.model')
    sdn_light = LightSDN(path_model_cnn, path_model_ics, sdn_type, num_classes, _device)

    if print_messages:
        print(f'[info] loading light SDN took {now() - t}')

    # step a)
    if print_messages:
        print('[info] computing confusion distributions')
    t = now()
    confusion_clean     = mf.compute_confusion(sdn_light, dataset_clean.train_loader,     _device)
    confusion_polygon   = mf.compute_confusion(sdn_light, dataset_polygon.train_loader,   _device)
    confusion_gotham    = mf.compute_confusion(sdn_light, dataset_gotham.train_loader,    _device)
    confusion_kelvin    = mf.compute_confusion(sdn_light, dataset_kelvin.train_loader,    _device)
    confusion_lomo      = mf.compute_confusion(sdn_light, dataset_lomo.train_loader,      _device)
    confusion_nashville = mf.compute_confusion(sdn_light, dataset_nashville.train_loader, _device)
    confusion_toaster   = mf.compute_confusion(sdn_light, dataset_toaster.train_loader,   _device)
    if print_messages:
        print(f'[info] computing confusion distribution for clean, polygon and filters took {now() - t}')
    # step b)
    if print_messages:
        print('[info] computing mean and std diffs')
    t = now()
    # with 0, 0 computes the plain mean and std
    clean_mean,          clean_std          = get_mean_std_diffs(confusion_clean,              0,         0, use_abs=use_abs_for_diff_features)
    mean_diff_polygon,   std_diff_polygon   = get_mean_std_diffs(confusion_polygon,   clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_gotham,    std_diff_gotham    = get_mean_std_diffs(confusion_gotham,    clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_kelvin,    std_diff_kelvin    = get_mean_std_diffs(confusion_kelvin,    clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_lomo,      std_diff_lomo      = get_mean_std_diffs(confusion_lomo,      clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_nashville, std_diff_nashville = get_mean_std_diffs(confusion_nashville, clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_toaster,   std_diff_toaster   = get_mean_std_diffs(confusion_toaster,   clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    if print_messages:
        print(f'[info] computing diff features took {now() - t}')

    features = np.array([
        mean_diff_polygon,   std_diff_polygon,
        mean_diff_gotham,    std_diff_gotham,
        mean_diff_kelvin,    std_diff_kelvin,
        mean_diff_lomo,      std_diff_lomo,
        mean_diff_nashville, std_diff_nashville,
        mean_diff_toaster,   std_diff_toaster
    ]).reshape(1, -1)

    if print_messages:
        print(f'[info] computed features (mean_diff, std_diff):')
        print('polygon:', mean_diff_polygon, std_diff_polygon)
        print('gotham:', mean_diff_gotham, std_diff_gotham)
        print('kelvin:', mean_diff_kelvin, std_diff_kelvin)
        print('lomo:', mean_diff_lomo, std_diff_lomo)
        print('nashville:', mean_diff_nashville, std_diff_nashville)
        print('toaster:', mean_diff_toaster, std_diff_toaster)
        print('all:', features)

    ################################################################################
    #################### STEP 4: predict backdoor probability
    ################################################################################
    if print_messages:
        print()
        print('[info] STEP 4: predicting backd proba')

    meta_model = af.load_obj(filename=path_meta_model)
    probabilities = meta_model.predict_proba(features)
    backd_proba = probabilities[0][1]

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
    parser.add_argument('--model_filepath',   type=str, default='./model.pt', help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath',  type=str, default='./output',   help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath',  type=str, default='./scratch',  help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, default='./example',  help='File path to the folder of examples which might be useful for determining whether a model is poisoned.')

    args = parser.parse_args()
    trojan_detector_umd(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)

# TODO: set a limit for the number of images per class when reading them from disk (Sanghyun's idea with 1,3,5 images per class)
# sudo singularity build umd_pipeline.simg umd_pipeline.def
# sudo singularity run --nv umd_pipeline.simg --model_filepath /home/ionut/trojai/TrojAI-test/id-1100/model.pt --result_filepath /home/ionut/trojai/TrojAI-test/id-1100_result.txt --scratch_dirpath /home/ionut/trojai/TrojAI-test/id-1100_scratch --examples_dirpath /home/ionut/trojai/TrojAI-test/id-1100/example_data
# sudo singularity run --nv umd_pipeline.simg --model_filepath ../TrojAI-test/id-1100/model.pt --result_filepath ../TrojAI-test/id-1100_result.txt --scratch_dirpath ../TrojAI-test/id-1100_scratch --examples_dirpath ../TrojAI-test/id-1100/example_data
# sudo singularity run --nv umd_pipeline.simg --model_filepath /id-1100/model.pt --result_filepath /id-1100_result.txt --scratch_dirpath /id-1100_scratch --examples_dirpath /id-1100/example_data
# py38 -W ignore umd_pipeline.py --model_filepath ../TrojAI-data/round2-train-dataset/id-00000000/model.pt --result_filepath ../TrojAI-data/round2-train-dataset/id-00000000/scratch/result.txt --scratch_dirpath TrojAI-data/round2-train-dataset/id-00000000/scratch --examples_dirpath ../TrojAI-data/round2-train-dataset/id-00000000/example_data
