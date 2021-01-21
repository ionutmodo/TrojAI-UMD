import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import ast
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor as Pool
import tools.model_funcs as mf
from architectures.LightSDN import LightSDN
from tools.network_architectures import load_trojai_model
from tools.logistics import *
from tools.logger import Logger
from notebooks.methods import encode_architecture, encode_backdoor
import synthetic_data.gen_backdoored_datasets as synthetic_module
import synthetic_data.aux_funcs as sdaf


def get_trigger_type_aux_value(triggers_0_type, triggers_0_instagram_filter_type, triggers_1_type, triggers_1_instagram_filter_type):
    triggers_0_type = triggers_0_type.lower()
    triggers_1_type = triggers_1_type.lower()
    triggers_0_instagram_filter_type = triggers_0_instagram_filter_type.lower().replace('filter', '').replace('xform', '')
    triggers_1_instagram_filter_type = triggers_1_instagram_filter_type.lower().replace('filter', '').replace('xform', '')

    if triggers_0_type == 'instagram':
        backd_0_str = f'instagram-{triggers_0_instagram_filter_type}'
        backd_0_code = encode_backdoor(triggers_0_instagram_filter_type)
    else:
        backd_0_str = triggers_0_type
        backd_0_code = encode_backdoor(triggers_0_type)

    if triggers_1_type == 'instagram':
        backd_1_str = f'instagram-{triggers_1_instagram_filter_type}'
        backd_1_code = encode_backdoor(triggers_1_instagram_filter_type)
    else:
        backd_1_str = triggers_1_type
        backd_1_code = encode_backdoor(triggers_1_type)

    return f'{backd_0_str}_{backd_1_str}', backd_0_code, backd_1_code


def worker_confusion_distribution(params):
    dict_params_dataset, dict_params_model = params
    _device = dict_params_model['device']
    dataset = TrojAI(**dict_params_dataset)
    sdn_light = LightSDN(**dict_params_model)

    dataset_name = os.path.basename(dict_params_dataset['folder'])
    conf_dist = mf.compute_confusion(sdn_light, dataset.train_loader, _device)

    return dataset_name, conf_dist


def main():
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
    list_limits = {
        'windows10': (0, 1007),
        'openlab08.umiacs.umd.edu': (0, 1007),
        # 'openlab30.umiacs.umd.edu': (0, 249),
        # 'openlab31.umiacs.umd.edu': (250, 499),
        # 'openlab32.umiacs.umd.edu': (500, 749),
        # 'openlab33.umiacs.umd.edu': (750, 1007),
    }

    if len(sys.argv) != 3:
        lim_left, lim_right = list_limits[socket.gethostname()]
    else:
        lim_left, lim_right = int(sys.argv[1]), int(sys.argv[2])

    print(f'lim_left={lim_left}, lim_right={lim_right}')

    # _device = 'cpu'
    _device = af.get_pytorch_device()

    test_ratio = 0
    batch_size = 128  # for confusion experiment

    experiment_name = f'fc_synthetic_polygon-all-gray_filters_{lim_left}-{lim_right}'
    sdn_name = 'ics_synthetic-1000_train100_test0_bs20'

    # begin
    np.random.seed(666)
    path_root_project = get_project_root_path()
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round2-train-dataset')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round2-holdout-dataset')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round3-train-dataset')
    # path_root = os.path.join(path_root_project, 'TrojAI-data', 'round3-holdout-dataset')
    path_root = os.path.join(path_root_project, 'TrojAI-data', 'round4-train-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')

    path_stats = os.path.join(path_root, 'synthetic_stats')

    # path_report_conf_dist = os.path.join(path_root, 'ics_svm', f'{os.path.basename(path_root)}_{experiment_name}.csv')
    path_report_conf_dist = os.path.join(path_root, 'ics_fc', f'{os.path.basename(path_root)}_{experiment_name}.csv')

    # path_logger = os.path.join(path_root, 'ics_svm', f'{os.path.basename(path_root)}_{experiment_name}.log')
    path_logger = os.path.join(path_root, 'ics_fc', f'{os.path.basename(path_root)}_{experiment_name}.log')

    Logger.open(path_logger)
    metadata = pd.read_csv(path_metadata)

    ############################################
    ########## LOAD SYNTHETIC DATASET ##########
    synthetic_data = np.load('synthetic_data/synthetic_data_1000_clean_polygon_instagram.npz')

    # continue training from where we left off last run
    if os.path.isfile(path_report_conf_dist):
        df_report_conf_dist = pd.read_csv(path_report_conf_dist)
        n_report_conf_dist = len(df_report_conf_dist)
        last_model_name_in_report_conf_dist = df_report_conf_dist.iloc[-1]['model_name']
        print(f'Continue training (last id is {last_model_name_in_report_conf_dist})')
    else:
        print('Training from scratch')
        last_model_name_in_report_conf_dist = None
        n_report_conf_dist = 0
        df_report_conf_dist = pd.DataFrame(columns=[
            # preliminary info about the model
            'model_name', 'model_architecture', 'architecture_code', 'backdoor_code_0', 'backdoor_code_1', 'model_label', 'backdoor_string',

            ## place differences first to visualize them easier
            # 'square30_mean_diff', 'square30_std_diff',
            # 'square30_r_mean_diff', 'square30_r_std_diff',
            'polygon_mean_diff', 'polygon_std_diff',
            'gotham_mean_diff', 'gotham_std_diff',
            'kelvin_mean_diff', 'kelvin_std_diff',
            'lomo_mean_diff', 'lomo_std_diff',
            'nashville_mean_diff', 'nashville_std_diff',
            'toaster_mean_diff', 'toaster_std_diff',

            # place effective metrics from confusion distribution
            'clean_mean', 'clean_std',
            # 'square30_mean', 'square30_std',
            # 'square30_r_mean', 'square30_r_std',
            'polygon_mean', 'polygon_std',
            'gotham_mean', 'gotham_std',
            'kelvin_mean', 'kelvin_std',
            'lomo_mean', 'lomo_std',
            'nashville_mean', 'nashville_std',
            'toaster_mean', 'toaster_std',

            # other data
            'num_classes'
        ])

    af.create_path(path_stats)
    # if not os.path.isdir(path_stats):
    #     os.makedirs(path_stats)

    for index, row in metadata.iterrows():
        start_time = datetime.now()
        model_name = row['model_name']
        model_id = int(model_name[3:])
        if lim_left <= model_id <= lim_right and ((last_model_name_in_report_conf_dist is None) or (last_model_name_in_report_conf_dist is not None and model_name > last_model_name_in_report_conf_dist)):
            backd_str, backd_0_code, backd_1_code = get_trigger_type_aux_value(row['triggers_0_type'], row['triggers_0_instagram_filter_type'], row['triggers_1_type'], row['triggers_1_instagram_filter_type'])
            model_label = 'backdoor' if row['poisoned'] else 'clean'
            model_architecture = row['model_architecture']
            num_classes = row['number_classes']
            synth_labeling_params = dict(model_img_size=int(row['cnn_img_size_pixels']), temperature=3.0)
            architecture_code = encode_architecture(model_architecture)

            ###############################################################################################################

            af.create_path(os.path.join(path_stats, model_name))

            Logger.log()
            Logger.log(experiment_name)
            Logger.log(f'model {model_name} {model_architecture} ({model_label}: {backd_str} - {backd_0_code}:{backd_1_code})')
            path_model = os.path.join(path_root, model_name)

            sdn_type = [v for k, v in dict_arch_type.items() if model_architecture.startswith(k)][0]
            path_model_cnn = os.path.join(path_model, 'model.pt')
            # path_model_ics = os.path.join(path_model, 'svm', 'svm_models')
            path_model_ics = os.path.join(path_model, sdn_name)

            dataset_conf_dist = dict(clean=None, polygon_all=None, gotham=None, kelvin=None, lomo=None, nashville=None, toaster=None)

            if socket.gethostname() in ['openlab08.umiacs.umd.edu', 'windows10']:
                # sdn_light = LightSDN(path_model_cnn, path_model_ics, sdn_type, num_classes, _device) ## the SVM version
                sdn_light = load_trojai_model(sdn_path=os.path.join(path_model, sdn_name),
                                              cnn_path=os.path.join(path_model, 'model.pt'),
                                              num_classes=num_classes, sdn_type=sdn_type, device=_device)
                for dataset_name in dataset_conf_dist:
                    # path_data = os.path.join(path_model, dataset_name)
                    # dataset = TrojAI(folder=path_data, test_ratio=test_ratio, batch_size=batch_size, device=_device, opencv_format=False)

                    ### Computing confusion distribution does not require the true labels
                    ### Instead, the confusion matrix approach requires the true categorial labels (in this case, they are distilled from the original CNN)
                    images, labels = synthetic_module.return_model_data_and_labels(sdn_light.model.cnn_model, synth_labeling_params, synthetic_data[dataset_name])
                    data = sdaf.ManualData(sdaf.convert_to_pytorch_format(images), labels['cat'])

                    # trick: replace original train loader with the synthetic loader
                    synthetic_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)

                    Logger.log(f'computing confusion for {dataset_name}...', end='')
                    # dict_dataset_confusion[dataset_name] = mf.compute_confusion(sdn_light, synthetic_loader, _device)
                    dataset_conf_dist[dataset_name] = mf.compute_confusion_distribution_and_matrix(sdn_light, synthetic_loader, num_classes, _device, stats_save_path=os.path.join(path_stats, model_name, f'{model_name}_{dataset_name}.npz'))
                    Logger.log('done')

                    del images, labels, data, synthetic_loader

                del sdn_light
            else:
                Logger.log(f'computing confusion for all datasets...', end='')
                mp_mapping_params = []
                for dataset_name in dataset_conf_dist:
                    path_data = os.path.join(path_model, dataset_name)
                    dict_params_dataset = dict(folder=path_data, test_ratio=test_ratio, batch_size=batch_size, device=_device, opencv_format=False)
                    dict_params_model = dict(path_model_cnn=path_model_cnn, path_model_ics=path_model_ics, sdn_type=sdn_type, num_classes=num_classes, device=_device)
                    mp_mapping_params.append((dict_params_dataset, dict_params_model))
                with Pool(len(mp_mapping_params)) as pool:
                    mp_result = pool.map(worker_confusion_distribution, mp_mapping_params)
                Logger.log(f'done')
                dataset_conf_dist = dict(mp_result)

            # compute mean and stds for confusion distributions
            clean_mean = np.mean(dataset_conf_dist['clean'])
            clean_std = np.std(dataset_conf_dist['clean'])

            # square30_mean = np.mean(dict_dataset_confusion['backdoored_data_square-30'])
            # square30_std = np.std(dict_dataset_confusion['backdoored_data_square-30'])
            
            polygon_mean = np.mean(dataset_conf_dist['polygon_all'])
            polygon_std = np.std(dataset_conf_dist['polygon_all'])

            gotham_mean = np.mean(dataset_conf_dist['gotham'])
            gotham_std = np.std(dataset_conf_dist['gotham'])

            kelvin_mean = np.mean(dataset_conf_dist['kelvin'])
            kelvin_std = np.std(dataset_conf_dist['kelvin'])

            lomo_mean = np.mean(dataset_conf_dist['lomo'])
            lomo_std = np.std(dataset_conf_dist['lomo'])

            nashville_mean = np.mean(dataset_conf_dist['nashville'])
            nashville_std = np.std(dataset_conf_dist['nashville'])

            toaster_mean = np.mean(dataset_conf_dist['toaster'])
            toaster_std = np.std(dataset_conf_dist['toaster'])

            ############ compute differences for mean and stds between backdoored and clean

            # square30_mean_diff = square30_mean - clean_mean
            # square30_std_diff = square30_std - clean_std

            # square30_r_mean_diff = square30_r_mean - clean_mean
            # square30_r_std_diff = square30_r_std - clean_std

            polygon_mean_diff = polygon_mean - clean_mean
            polygon_std_diff = polygon_std - clean_std

            gotham_mean_diff = gotham_mean - clean_mean
            gotham_std_diff = gotham_std - clean_std

            kelvin_mean_diff = kelvin_mean - clean_mean
            kelvin_std_diff = kelvin_std - clean_std

            lomo_mean_diff = lomo_mean - clean_mean
            lomo_std_diff = lomo_std - clean_std

            nashville_mean_diff = nashville_mean - clean_mean
            nashville_std_diff = nashville_std - clean_std

            toaster_mean_diff = toaster_mean - clean_mean
            toaster_std_diff = toaster_std - clean_std

            df_report_conf_dist.loc[n_report_conf_dist] = [
                # preliminary info about the model
                model_name, model_architecture, architecture_code, backd_0_code, backd_1_code, model_label, backd_str,

                # square30_mean_diff, square30_std_diff,
                # square30_r_mean_diff, square30_r_std_diff,

                polygon_mean_diff, polygon_std_diff,
                gotham_mean_diff, gotham_std_diff,
                kelvin_mean_diff, kelvin_std_diff,
                lomo_mean_diff, lomo_std_diff,
                nashville_mean_diff, nashville_std_diff,
                toaster_mean_diff, toaster_std_diff,

                ## place effective metrics from confusion distribution
                clean_mean, clean_std,
                # square30_mean, square30_std,
                # square30_r_mean, square30_r_std,
                polygon_mean, polygon_std,
                gotham_mean, gotham_std,
                kelvin_mean, kelvin_std,
                lomo_mean, lomo_std,
                nashville_mean, nashville_std,
                toaster_mean, toaster_std,

                # other data
                num_classes
            ]
            n_report_conf_dist += 1
            df_report_conf_dist.to_csv(path_report_conf_dist, index=False)
            end_time = datetime.now()

            del dataset_conf_dist
            torch.cuda.empty_cache()

            Logger.log(f'model {model_name} took {end_time - start_time}')
    Logger.close()


if __name__ == '__main__':
    main()
