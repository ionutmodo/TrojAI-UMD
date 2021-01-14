import os
import numpy as np
import torch
import pandas

import warnings 
warnings.filterwarnings("ignore")

import csv
import synthetic_data.aux_funcs as af
import synthetic_data.model_funcs as mf
import synthetic_data.InternalClassifiers as ic
import synthetic_data.gen_backdoored_datasets as bd
from tools.aux_funcs import save_obj

import pickle


def generate_sdn_model(model, loaders, model_architecture, model_input_size, num_classes, save_path, model_type):
    collect_all_layers = True if 'all' in model_type else False

    # ics exist
    if os.path.isfile(save_path):
        print('The ICs {} exists.'.format(save_path))
        ics = af.load_model(save_path, model_type)
        layers_to_collect = 'all' if collect_all_layers else mf.get_layer_hook_names(model_architecture)
        activation_extractor = mf.ActivationExtractor(model, layers=layers_to_collect)
        ics.add_activation_extractor(activation_extractor)

    else:    
        # collect the activation shapes of the current model to create the ICs
        layers_to_collect = 'all' if collect_all_layers else mf.get_layer_hook_names(model_architecture)
        activation_extractor = mf.ActivationExtractor(model, layers=layers_to_collect).eval()
        dummy_input = torch.from_numpy(np.random.rand(1, 3, 224, 224)).float().cuda()
        features, _ = activation_extractor(dummy_input)
        ic_input_shapes = [output.shape[1] for output in features] # num filters
        
        # print('Internal Layer Activation Shapes for ICs: {}'.format(ic_input_shapes))

        if 'fcn' in model_type:
            print('Training FCN ICs...')
            # define the IC model
            structure_params = []
            ics = ic.InternalClassifiers(ic_input_shapes, num_classes, structure_params).cuda()
            ics.add_activation_extractor(activation_extractor)
            # train the ICs
            optim_params = {'optim_type': 'adam', 'init_lr': 0.0001, 'reduce_lr_epochs': [20], 'reduce_lr_factors': [0.1], 'weight_decay': 1e-5, 'epochs': 30}
            ic.train_ics(ics, (loaders['train'], loaders['test']), optim_params)
        
        else:
            params = {}
            params['num_neighbors'] =  int(model_type.split('_')[1]) if 'knn' in model_type else 0 # parameters for KNN
            params['kernel'], params['gamma'], params['C'] = 'linear', 'scale', 1 # parameters for SVR

            simple_model_type = model_type.split('_')[0]

            print('Training Simple IC {}...'.format(model_type))
            ics = ic.InternalClassifiers_Simple(ic_input_shapes, num_classes, simple_model_type, params)
            ics.add_activation_extractor(activation_extractor)
            # train the ICs
            ics.train_models(loaders['train'])
            
    # test the ics
    print('Testing the final model...')

    if 'real_test' in loaders:
        accs = ic.ics_test(ics, loaders['real_test'])
        print('Layerwise ICs Real Test Accs: {}'.format(accs))

    if 'real_train' in loaders:
        accs = ic.ics_test(ics, loaders['real_train'])
        print('Layerwise ICs Real Train Accs: {}'.format(accs))

    if 'syn_train' in loaders:
        accs = ic.ics_test(ics, loaders['syn_train'])
        print('Layerwise ICs Sythetic Train Accs: {}'.format(accs))

    if 'syn_test' in loaders:
        accs = ic.ics_test(ics, loaders['syn_test'])
        print('Layerwise ICs Sythetic Test Accs: {}'.format(accs))

    if 'syn_trigger' in loaders:
        accs = ic.ics_test(ics, loaders['syn_trigger'])
        print('Layerwise ICs Sythetic Triggered Accs: {}'.format(accs))

    if 'real_trigger' in loaders:
        accs = ic.ics_test(ics, loaders['real_trigger'])
        print('Layerwise ICs Sythetic Triggered Accs: {}'.format(accs))

    if not os.path.isfile(save_path):
        print('Saving the ICs...')
        # save the ICs
        ics.remove_activation_extractor()
        activation_extractor.remove_hooks()
        af.save_model(ics, save_path, model_type)

    return ics


def save_synthetic_data(img_size, img_count, trigger_types, foregrounds_path, backgrounds_path, triggers_path, synthetic_data_save_path):
    params = {}

    params['img_size'] = img_size

    # for the triggered dataset, these triggers will be added to synthetic clean images
    params['trigger_types'] = trigger_types
    params['foregrounds_path'] =  foregrounds_path
    params['backgrounds_path'] = backgrounds_path
    params['triggers_path'] = triggers_path

    # these are for the polygon triggers
    params['foregrounds_min'], params['foregrounds_max'] = (0.3, 0.6) # area ratio
    params['trigger_size_min'], params['trigger_size_max'] = (0.05, 0.2) # area ratio
    params['trigger_color'] = (127, 127, 127) # RGB
    params['trigger_side_count'] = 'all' # params['trigger_side_count'] = 3

    # this many clean synthetic images will be created, and for each trigger type num_images triggered images will be created
    params['num_images'] = img_count

    # how to label the synthetic images using the model's predictions, 'cat' is categorical label (label 0, label 1), and 'soft' is the prob. distributions for each image
    params['label_type'] = 'cat'


    # create synthetic clean data and triggered data
    # clean_data, triggered_data  = bd.create_synthetic_dataset(params)
    # np.savez_compressed(synthetic_data_save_path, clean_data=clean_data, triggered_data=triggered_data)

    synthetic_images_dict = bd.create_synthetic_dataset(params)
    np.savez_compressed(f'{synthetic_data_save_path}.npz', **synthetic_images_dict)
    save_obj(synthetic_images_dict, f'{synthetic_data_save_path}.pkl')


def train_ics_w_synthetic_data(model_type, use_sythetic, label_type, synthetic_data_path):
    metadata_file = 'METADATA.csv'
    
    models_path = 'models'
    
    batch_size = 16
    train_shuffle = False
    num_workers = 4

    df = pandas.read_csv(metadata_file)

    for model_idx, model_name in enumerate(df['model_name']):
        cur_path = os.path.join(models_path, model_name)
        if not os.path.isdir(cur_path):
            continue
        
        arch = df['model_architecture'][model_idx]
        poisoned = df['poisoned'][model_idx]
        
        trigger_type = df['trigger_type'][model_idx]
        trigerred_classes = df['triggered_classes'][model_idx]
        input_size = df['cnn_img_size_pixels'][model_idx]
        num_classes = df['number_classes'][model_idx]
        filter_type = df['instagram_filter_type'][model_idx]

        print('Model Name: {} - Architecture: {} - Poisoned: {} - Num Classes: {} - Trigger Type: {} - Triggered Classes: {} - Filter : {}'.format(model_name, arch, poisoned, num_classes, trigger_type, trigerred_classes, filter_type))

        model_filepath = os.path.join(cur_path, 'model.pt')
        model = torch.load(model_filepath, map_location=torch.device('cuda')).eval()

        loaders = {}
        # if the model is poisoned, load the provided triggered images for testing
        if poisoned:
            trigger_examples_dirpath = os.path.join(cur_path, 'poisoned_example_data')
            real_triggered_loader = af.get_loader(af.get_img_data(trigger_examples_dirpath, input_size, test_size=0, example_img_format='png'))
            loaders['real_trigger'] = real_triggered_loader

        if use_sythetic:
            print('Using synthetic data to train ICs.')
            params = {}
            params['model_img_size'] = int(input_size)
            params['temperature'] = 5
            # load the synthetic data
            data = np.load(synthetic_data_path)
            clean_data, triggered_data = data['clean_data'], data['triggered_data']

            # label the synthetic data
            clean_data, labels = bd.return_model_data_and_labels(model, params, clean_data)
            triggered_data, _ = bd.return_model_data_and_labels(model, params, triggered_data)

            clean_labels = labels[label_type]
            triggered_labels = np.asarray(labels['cat'].tolist() * int(len(triggered_data)/len(clean_data))).astype(int)

            clean_data = af.ManualData(af.convert_to_pytorch_format(clean_data), clean_labels)
            triggered_data = af.ManualData(af.convert_to_pytorch_format(triggered_data), triggered_labels)

            sythetic_train_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
            synthetic_triggered_loader = torch.utils.data.DataLoader(triggered_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # use the real examples provided by TrojAI for testing
            examples_dirpath = os.path.join(cur_path, 'clean_example_data')
            real_test_loader = af.get_loader(af.get_img_data(examples_dirpath, input_size, test_size=0, example_img_format='png'))
            loaders['train'] = loaders['syn_train'] = sythetic_train_loader
            loaders['test'] = loaders['real_test'] = real_test_loader
            loaders['syn_trigger'] = synthetic_triggered_loader
            suffix = 'synthetic'

        else:
            print('Using real data to train ICs.')
            # use the real examples provided by TrojAI for training and testing
            examples_dirpath = os.path.join(cur_path, 'clean_example_data')
            real_train_data, real_test_data = af.get_img_data(examples_dirpath, input_size, test_size=0.2, example_img_format='png')
            real_train_loader, real_test_loader = af.get_loader(real_train_data), af.get_loader(real_test_data)
            loaders['train'] = loaders['real_train'] = real_train_loader
            loaders['test'] = loaders['real_test'] = real_test_loader
            suffix = 'real'
        
        ics_save_path = os.path.join(cur_path, 'ics_{}_{}.dat'.format(suffix, model_type))
        generate_sdn_model(model, loaders, arch, input_size, num_classes, ics_save_path, model_type)

        print('======================================')


if __name__ == "__main__":
    af.set_logger('deneme.log')
    synthetic_data_path = f'../../TrojAI-SyntheticDataset'

    # Generate synthetic data and save it to the disk
    trigger_types = ['polygon_all', 'gotham', 'kelvin', 'lomo', 'nashville', 'toaster']
    backgrounds_path = os.path.join(synthetic_data_path, 'backgrounds')
    triggers_path = os.path.join(synthetic_data_path, 'triggers')
    foregrounds_path = os.path.join(synthetic_data_path, 'foregrounds')
    img_size = 256
    img_count = 100
    synthetic_data_save_path = f'synthetic_data_{img_count}_clean_polygon_instagram'
    save_synthetic_data(img_size, img_count, trigger_types, foregrounds_path, backgrounds_path, triggers_path, synthetic_data_save_path)


    # model_type = 'knn_12' # uses knn ICs with k=5 by collecting the activations defined in model_funcs.get_layer_hook_names
    # model_type = 'knn_5_all' # uses knn ICs with k=5 by collecting the activations from all conv layers
    # model_type = 'fcn' # uses a fully connected neural network as ICs by collecting the activations defined in model_funcs.get_layer_hook_names
    # model_type = 'fcn_all' # uses a fully connected neural network as ICs by collecting the activations from all conv layers - this is too huge, don't use this.
    # model_type = 'gbr'

    # model_type = 'svr'
    #
    # label_type = 'soft'
    # #label_type = 'cat'
    #
    # use_sythetic = True
    # # use_sythetic = False
    #
    # train_ics_w_synthetic_data(model_type, use_sythetic, label_type, synthetic_data_save_path)
