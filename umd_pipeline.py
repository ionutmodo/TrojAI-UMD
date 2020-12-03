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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow messages
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
from architectures.MLP import LayerwiseClassifiers
from tools.network_architectures import load_trojai_model, save_model
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
from keras.models import model_from_json
from concurrent.futures import ProcessPoolExecutor as Pool


def keras_save(model, folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    model_json = model.to_json()
    with open(os.path.join(folder, 'model.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(folder, 'model.h5'))


def keras_load(folder):

    json_file = open(os.path.join(folder, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(folder, 'model.h5'))
    return loaded_model


def now():
    return datetime.now()


def write_prediction(filepath, backd_proba):
    with open(filepath, 'w') as w:
        w.write(backd_proba)


def worker_backdoored_dataset_creator(params):
    create_backdoored_dataset(**params)


def parallelize_backdoored_dataset_creation(p_examples_dirpath, p_scratch_dirpath, p_trigger_size, p_trigger_color, p_trigger_target_class, p_list_filters):
    print('[info] parallelizing...')
    mp_mapping_params = [dict(dir_clean_data=p_examples_dirpath,
                              dir_backdoored_data=os.path.join(p_scratch_dirpath, f'backdoored_data_square_{p_trigger_size}'),
                              trigger_type='polygon',
                              trigger_name='square',
                              trigger_color=p_trigger_color,
                              trigger_size=p_trigger_size,
                              triggered_classes='all',
                              trigger_target_class=p_trigger_target_class)]

    # create filters dataset and save it to disk
    for p_filter in p_list_filters:
        mp_mapping_params.append(dict(dir_clean_data=p_examples_dirpath,
                                      dir_backdoored_data=os.path.join(p_scratch_dirpath, f'backdoored_data_filter_{p_filter}'),
                                      trigger_type='filter',
                                      trigger_name=p_filter,
                                      trigger_color=None,
                                      trigger_size=None,
                                      triggered_classes='all',
                                      trigger_target_class=p_trigger_target_class))

    with Pool(max_workers=len(mp_mapping_params)) as pool:
        pool.map(worker_backdoored_dataset_creator, mp_mapping_params)


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
        # if log:
        #     y_pred = svm.predict(features[i])
        #
        #     list_raw_acc = []
        #     list_bal_acc = []
        #     for c in range(n_classes):
        #         acc_raw = accuracy_score(y_true=labels[:, c], y_pred=y_pred[:, c])
        #         acc_balanced = balanced_accuracy_score(y_true=labels[:, c], y_pred=y_pred[:, c])
        #
        #         list_raw_acc.append(f'{acc_raw * 100.0:.2f}')
        #         list_bal_acc.append(f'{acc_balanced * 100.0:.2f}')
        #
        #     print(f'[info] SVM-IC-{i} Raw Acc: [{", ".join(list_raw_acc)}]')
        #     print(f'[info] SVM-IC-{i} Bal Acc: [{", ".join(list_bal_acc)}]')
        #     print(f'--------------------------------------------------------------------------')

    path_svm_model = os.path.join(model_root_path, 'ics_svm.model')
    af.save_obj(obj=svm_ics, filename=path_svm_model)
    size = os.path.getsize(path_svm_model) / (2 ** 20)
    if log:
        Logger.log(f'[info] SVM model ({size:.2f} MB) saved to {path_svm_model}')


def train_trojai_sdn(dataset, trojai_model_w_ics, model_root_path, device):
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
    save_model(ics, params, model_root_path, f'ics_train{train_proc}_test{test_proc}_bs{bs}', epoch=-1)


def trojan_detector_umd(model_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    time_start = datetime.now()
    print_messages = True
    trigger_size = 30 # for polygon dataset
    trigger_color = 'random'
    # trigger_color = (127, 127, 127)
    trigger_target_class = 0
    list_filters = ['gotham', 'kelvin', 'lomo', 'nashville', 'toaster']

    path_meta_model = 'metamodels/metamodel_10_fc_round3_data=diffs_square=30-rand_scaler=STD_clf=LR-1'

    batch_size_sdn_training = 10 if socket.gethostname() == 'windows10' else 20
    batch_size = 1 if socket.gethostname() == 'windows10' else 50
    _device = af.get_pytorch_device()
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
    # the batch_size=20 is only to train the SDNs !!!
    dataset_clean, sdn_type, model = read_model_directory(model_filepath, examples_dirpath, batch_size=batch_size_sdn_training, test_ratio=0, device=_device)
    if print_messages:
        print(f'[info] reading clean dataset and raw model took {now() - t}')
    num_classes = dataset_clean.num_classes

    if print_messages:
        print()
        print('[info] STEP 1: train SDN with SVM ICs')
    # this method saves the SVM-ICs to "scratch_dirpath"/svm/svm_models (the file has no extension)
    t = now()
    ###### train_trojai_sdn_with_svm(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=_device, log=print_messages)
    train_trojai_sdn(dataset=dataset_clean, trojai_model_w_ics=model, model_root_path=scratch_dirpath, device=_device)
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
    create_backdoored_dataset(dir_clean_data=examples_dirpath,
                              dir_backdoored_data=os.path.join(scratch_dirpath, f'backdoored_data_square_{trigger_size}'),
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
    if print_messages:
        print(f'[info] creating all backdoored datasets took {now() - t}')

    ################################################################################
    #################### STEP 3: create backdoored datasets
    ################################################################################
    # create paths
    path_polygon   = os.path.join(scratch_dirpath, f'backdoored_data_square_{trigger_size}')
    path_gotham    = os.path.join(scratch_dirpath, f'backdoored_data_filter_gotham')
    path_kelvin    = os.path.join(scratch_dirpath, f'backdoored_data_filter_kelvin')
    path_lomo      = os.path.join(scratch_dirpath, f'backdoored_data_filter_lomo')
    path_nashville = os.path.join(scratch_dirpath, f'backdoored_data_filter_nashville')
    path_toaster   = os.path.join(scratch_dirpath, f'backdoored_data_filter_toaster')

    if print_messages:
        print()
        print('[info] STEP 3: loading backdoored datasets, computing confusion distribution')

    # load model
    t = now()
    # path_model_cnn = model_filepath
    # path_model_ics = os.path.join(scratch_dirpath, 'ics_svm.model')
    # sdn_light = LightSDN(path_model_cnn, path_model_ics, sdn_type, num_classes, _device)
    sdn_light = load_trojai_model(sdn_path=os.path.join(scratch_dirpath, sdn_name),
                                  cnn_path=model_filepath,
                                  num_classes=num_classes, sdn_type=sdn_type, device=_device)

    if print_messages:
        print(f'[info] loading light SDN took {now() - t}')

    t = now()

    dataset_clean = TrojAI(folder=examples_dirpath, test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    confusion_clean = mf.compute_confusion(sdn_light, dataset_clean.train_loader, _device)
    # confusion_clean = [0, 0]
    mean_clean, std_clean = np.mean(confusion_clean), np.std(confusion_clean)
    del dataset_clean, confusion_clean

    dataset_polygon = TrojAI(folder=path_polygon,   test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    confusion_polygon = mf.compute_confusion(sdn_light, dataset_polygon.train_loader, _device)
    # confusion_polygon = [0, 0]
    mean_polygon, std_polygon = np.mean(confusion_polygon), np.std(confusion_polygon)
    mean_diff_polygon, std_diff_polygon = abs(mean_polygon - mean_clean), abs(std_polygon - std_clean)
    del dataset_polygon, confusion_polygon

    dataset_gotham = TrojAI(folder=path_gotham, test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    confusion_gotham = mf.compute_confusion(sdn_light, dataset_gotham.train_loader, _device)
    # confusion_gotham = [0, 0]
    mean_gotham, std_gotham = np.mean(confusion_gotham), np.std(confusion_gotham)
    mean_diff_gotham, std_diff_gotham = abs(mean_gotham - mean_clean), abs(std_gotham - std_clean)
    del dataset_gotham, confusion_gotham

    dataset_kelvin = TrojAI(folder=path_kelvin, test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    confusion_kelvin = mf.compute_confusion(sdn_light, dataset_kelvin.train_loader, _device)
    # confusion_kelvin = [0, 0]
    mean_kelvin, std_kelvin = np.mean(confusion_kelvin), np.std(confusion_kelvin)
    mean_diff_kelvin, std_diff_kelvin = abs(mean_kelvin - mean_clean), abs(std_kelvin - std_clean)
    del dataset_kelvin, confusion_kelvin

    dataset_lomo = TrojAI(folder=path_lomo, test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    confusion_lomo = mf.compute_confusion(sdn_light, dataset_lomo.train_loader, _device)
    # confusion_lomo = [0, 0]
    mean_lomo, std_lomo = np.mean(confusion_lomo), np.std(confusion_lomo)
    mean_diff_lomo, std_diff_lomo = abs(mean_lomo - mean_clean), abs(std_lomo - std_clean)
    del dataset_lomo, confusion_lomo

    dataset_nashville = TrojAI(folder=path_nashville, test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    confusion_nashville = mf.compute_confusion(sdn_light, dataset_nashville.train_loader, _device)
    # confusion_nashville = [0, 0]
    mean_nashville, std_nashville = np.mean(confusion_nashville), np.std(confusion_nashville)
    mean_diff_nashville, std_diff_nashville = abs(mean_nashville - mean_clean), abs(std_nashville - std_clean)
    del dataset_nashville, confusion_nashville

    dataset_toaster = TrojAI(folder=path_toaster,   test_ratio=0, batch_size=batch_size, device=_device, opencv_format=False)
    confusion_toaster = mf.compute_confusion(sdn_light, dataset_toaster.train_loader,   _device)
    # confusion_toaster = [0, 0]
    mean_toaster, std_toaster = np.mean(confusion_toaster), np.std(confusion_toaster)
    mean_diff_toaster, std_diff_toaster = abs(mean_toaster - mean_clean), abs(std_toaster - std_clean)
    del dataset_toaster, confusion_toaster

    if print_messages:
        print()
        print(f'[info] computing confusion distribution for clean, polygon and filters took {now() - t}')

    ## DIFF FEATURES
    features_diff = np.array([
        mean_diff_polygon,   std_diff_polygon,
        mean_diff_gotham,    std_diff_gotham,
        mean_diff_kelvin,    std_diff_kelvin,
        mean_diff_lomo,      std_diff_lomo,
        mean_diff_nashville, std_diff_nashville,
        mean_diff_toaster,   std_diff_toaster
    ]).reshape(1, -1)

    ## RAW FEATURES
    features_raw = np.array([
        mean_clean,     std_clean,
        mean_polygon,   std_polygon,
        mean_gotham,    std_gotham,
        mean_kelvin,    std_kelvin,
        mean_lomo,      std_lomo,
        mean_nashville, std_nashville,
        mean_toaster,   std_toaster,
    ]).reshape(1, -1)

    ### SETTING FEATURES VARIABLE
    features = features_diff

    if print_messages:
        print(f'[info] computed features for model {model_filepath.split(os.path.sep)[-2]}')
        print('[raw  feature] clean:', mean_clean, std_clean)
        print('[raw  feature] polygon:', mean_polygon, std_polygon)
        print('[raw  feature] gotham:', mean_gotham, std_gotham)
        print('[raw  feature] kelvin:', mean_kelvin, std_kelvin)
        print('[raw  feature] lomo:', mean_lomo, std_lomo)
        print('[raw  feature] nashville:', mean_nashville, std_nashville)
        print('[raw  feature] toaster:', mean_toaster, std_toaster)
        print('[raw  feature] raw features:', features_raw.tolist())
        print()
        print('[diff feature] polygon:', mean_diff_polygon, std_diff_polygon)
        print('[diff feature] gotham:', mean_diff_gotham, std_diff_gotham)
        print('[diff feature] kelvin:', mean_diff_kelvin, std_diff_kelvin)
        print('[diff feature] lomo:', mean_diff_lomo, std_diff_lomo)
        print('[diff feature] nashville:', mean_diff_nashville, std_diff_nashville)
        print('[diff feature] toaster:', mean_diff_toaster, std_diff_toaster)
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

    meta_model = af.load_obj(filename=os.path.join(path_meta_model, 'model.pickle'))
    positive_class_index = np.where(meta_model.classes_ == 1)[0][0] # only for sklearn models
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
    parser.add_argument('--model_filepath',   type=str, default='./model.pt', help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath',  type=str, default='./output',   help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath',  type=str, default='./scratch',  help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, default='./example',  help='File path to the folder of examples which might be useful for determining whether a model is poisoned.')

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
