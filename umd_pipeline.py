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

from tools.logistics import *
from train_trojai_sdn import train_trojai_sdn_with_svm
from tools.data import create_backdoored_dataset
from architectures.SDNs.LightSDN import LightSDN
import tools.model_funcs as mf
import tools.aux_funcs as af
import numpy as np
import argparse


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


def trojan_detector_umd(model_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    use_abs_for_diff_features = False
    trigger_size = 20 # for polygon dataset
    trigger_color = (0, 0, 0) # also try (127, 127, 127) or random (R, G, B) files
    trigger_target_class = 0 # can be anything, its used just for the new file name
    list_filters = ['gotham', 'kelvin', 'lomo', 'nashville', 'toaster']
    path_meta_model = 'TO-BE-FILLED'
    batch_size = 50
    device = af.get_pytorch_device()

    ################################################################################
    #################### STEP 1: train SDN
    ################################################################################
    dataset_clean, sdn_type, model = read_model_directory(model_root=model_filepath, batch_size=batch_size, test_ratio=0, device=device)
    num_classes = dataset_clean.num_classes

    # this method saves the SVM-ICs to "scratch_dirpath"/svm/svm_models (the file has no extension)
    train_trojai_sdn_with_svm(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=device, log=False)

    ################################################################################
    #################### STEP 2: create backdoored datasets
    ################################################################################
    ### the speed can be improved by creating the datasets using multiprocessing (1 process for each dataset to be created)
    # create polygon dataset and save it to disk
    create_backdoored_dataset(dir_clean_data=examples_dirpath,
                              dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_polygon_{trigger_size}'),
                              trigger_type='polygon',
                              trigger_name='square',
                              trigger_color=trigger_color,
                              trigger_size=trigger_size,
                              triggered_classes='all',
                              trigger_target_class=trigger_target_class)
    # create filters dataset and save it to disk
    for p_filter in list_filters:
        create_backdoored_dataset(dir_clean_data=examples_dirpath,
                                  dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_filter_{p_filter}'),
                                  trigger_type='filter',
                                  trigger_name=p_filter,
                                  trigger_color=None,
                                  trigger_size=None,
                                  triggered_classes='all',
                                  trigger_target_class=trigger_target_class)

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

    # the clean dataset is loaded at the beginning
    dataset_polygon   = TrojAI(folder=path_polygon,   test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
    dataset_gotham    = TrojAI(folder=path_gotham,    test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
    dataset_kelvin    = TrojAI(folder=path_kelvin,    test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
    dataset_lomo      = TrojAI(folder=path_lomo,      test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
    dataset_nashville = TrojAI(folder=path_nashville, test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)
    dataset_toaster   = TrojAI(folder=path_toaster,   test_ratio=0, batch_size=batch_size, device=device, opencv_format=False)

    # load model
    path_model_cnn = model_filepath
    path_model_ics = os.path.join(scratch_dirpath, 'svm', 'svm_models')
    sdn_light = LightSDN(path_model_cnn, path_model_ics, sdn_type, num_classes, device)

    # step a)
    confusion_clean     = mf.compute_confusion(sdn_light, dataset_clean.train_loader,     device)
    confusion_polygon   = mf.compute_confusion(sdn_light, dataset_polygon.train_loader,   device)
    confusion_gotham    = mf.compute_confusion(sdn_light, dataset_gotham.train_loader,    device)
    confusion_kelvin    = mf.compute_confusion(sdn_light, dataset_kelvin.train_loader,    device)
    confusion_lomo      = mf.compute_confusion(sdn_light, dataset_lomo.train_loader,      device)
    confusion_nashville = mf.compute_confusion(sdn_light, dataset_nashville.train_loader, device)
    confusion_toaster   = mf.compute_confusion(sdn_light, dataset_toaster.train_loader,   device)

    # step b)
    clean_mean,          clean_std          = get_mean_std_diffs(confusion_clean, 0, 0, use_abs=use_abs_for_diff_features)  # with 0, 0 computes the plain mean and std
    mean_diff_polygon,   std_diff_polygon   = get_mean_std_diffs(confusion_polygon,   clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_gotham,    std_diff_gotham    = get_mean_std_diffs(confusion_gotham,    clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_kelvin,    std_diff_kelvin    = get_mean_std_diffs(confusion_kelvin,    clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_lomo,      std_diff_lomo      = get_mean_std_diffs(confusion_lomo,      clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_nashville, std_diff_nashville = get_mean_std_diffs(confusion_nashville, clean_mean, clean_std, use_abs=use_abs_for_diff_features)
    mean_diff_toaster,   std_diff_toaster   = get_mean_std_diffs(confusion_toaster,   clean_mean, clean_std, use_abs=use_abs_for_diff_features)

    features = np.array([
        mean_diff_polygon,   std_diff_polygon,
        mean_diff_gotham,    std_diff_gotham,
        mean_diff_kelvin,    std_diff_kelvin,
        mean_diff_lomo,      std_diff_lomo,
        mean_diff_nashville, std_diff_nashville,
        mean_diff_toaster,   std_diff_toaster
    ])

    ################################################################################
    #################### STEP 4: predict backdoor probability
    ################################################################################
    meta_model = af.load_obj(filename=path_meta_model)
    backd_proba = meta_model.predict_proba(features)

    ### write prediction to file
    write_prediction(result_filepath, backd_proba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath',   type=str, default='./model.pt', help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath',  type=str, default='./output',   help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath',  type=str, default='./scratch',  help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, default='./example',  help='File path to the folder of examples which might be useful for determining whether a model is poisoned.')

    args = parser.parse_args()
    trojan_detector_umd(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)

# TODO: set a limit for the number of images per class when reading them from disk
