import os
import socket

import numpy as np
import torch

import tools.aux_funcs as af
import tools.model_funcs as mf
import tools.network_architectures as arcs
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNDenseNet121 import SDNDenseNet121
from tools.data import TrojAI


def model_confusion_experiment(models_path, model_id, sdn_type, device='cpu'):
    suffix = 'train60_test40_bs25'
    sdn_name = f'ics_{suffix}'
    cnn_name = f'densenet121_{suffix}'

    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model = sdn_model.to(device)

    sdn_images = f'confusion_experiments/{model_id}_{sdn_name}'
    cnn_images = f'confusion_experiments/{model_id}_{cnn_name}'

    cnn_model = torch.load(os.path.join(models_path, 'model.pt')).to(device)
    cnn_model = SDNDenseNet121(cnn_model, (1, 3, 224, 224), 5, sdn_type, device)
    sdn_model.set_model(cnn_model)

    dataset = TrojAI(folder=os.path.join(models_path, 'example_data'), batch_size=10, device=device)

    # top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
    # print('SDN Top1 Test accuracy: {}'.format(top1_test))
    # print('SDN Top5 Test accuracy: {}'.format(top5_test))
    #
    # top1_test, top5_test = mf.cnn_test(cnn_model, dataset.test_loader, device)
    # print('CNN Top1 Test accuracy: {}'.format(top1_test))
    # print('CNN Top5 Test accuracy: {}'.format(top5_test))

    # the the normalization stats from the training set
    confusion_stats = mf.sdn_confusion_stats(sdn_model, loader=dataset.train_loader, device=device)
    print(f'Confusion stats: mean={confusion_stats[0]}, std={confusion_stats[1]}')

    sdn_layer_correct, sdn_layer_wrong, sdn_instance_confusion = mf.sdn_get_confusion(sdn_model, loader=dataset.test_loader, confusion_stats=confusion_stats,
                                                                                      device=device)
    sdn_correct_confusion, sdn_wrong_confusion = mf.get_sdn_stats(sdn_layer_correct, sdn_layer_wrong, sdn_instance_confusion)

    cnn_correct, cnn_wrong, cnn_instance_confidence = mf.cnn_get_confidence(cnn_model, loader=dataset.test_loader, device=device)
    cnn_correct_confidence, cnn_wrong_confidence = mf.get_cnn_stats(cnn_correct, cnn_wrong, cnn_instance_confidence)

    af.create_path(sdn_images)
    af.create_path(cnn_images)

    # corrects and wrongs
    af.overlay_two_histograms(sdn_images,
                              f'{model_id}_{sdn_name}_corrects_wrongs',
                              sdn_correct_confusion,
                              sdn_wrong_confusion,
                              'Correct',
                              'Wrong',
                              'SDN Confusion')
    af.overlay_two_histograms(cnn_images,
                              f'{model_id}_{cnn_name}_corrects_wrongs',
                              cnn_correct_confidence,
                              cnn_wrong_confidence,
                              'Correct',
                              'Wrong',
                              'CNN Confidence')

    mean_first = np.mean(sdn_correct_confusion)
    mean_second = np.mean(sdn_wrong_confusion)

    in_first_above_second = 0
    in_second_below_first = 0

    for item in sdn_correct_confusion:
        if float(item) > float(mean_second):
            in_first_above_second += 1

    for item in sdn_wrong_confusion:
        if float(item) < float(mean_first):
            in_second_below_first += 1

    print('SDN more confused correct: {}/{}'.format(in_first_above_second, len(sdn_correct_confusion)))
    print('SDN less confused wrong: {}/{}'.format(in_second_below_first, len(sdn_wrong_confusion)))

    mean_first = np.mean(cnn_correct_confidence)
    mean_second = np.mean(cnn_wrong_confidence)

    in_first_below_second = 0
    in_second_above_first = 0
    for item in cnn_correct_confidence:
        if float(item) < float(mean_second):
            in_first_below_second += 1
    for item in cnn_wrong_confidence:
        if float(item) > float(mean_first):
            in_second_above_first += 1

    print('CNN less confident correct: {}/{}'.format(in_first_below_second, len(cnn_correct_confidence)))
    print('CNN more confident wrong: {}/{}'.format(in_second_above_first, len(cnn_wrong_confidence)))

    # all instances
    sdn_all_confusion = list(sdn_instance_confusion.values())
    cnn_all_confidence = list(cnn_instance_confidence.values())

    print('Avg SDN Confusion: {}'.format(np.mean(sdn_all_confusion)))
    print('Std SDN Confusion: {}'.format(np.std(sdn_all_confusion)))

    print('Avg CNN Confidence: {}'.format(np.mean(cnn_all_confidence)))
    print('Std CNN Confidence {}'.format(np.std(cnn_all_confidence)))


if __name__ == '__main__':
    device = af.get_pytorch_device()

    hostname = socket.gethostname()
    hostname = 'openlab' if hostname.startswith('openlab') else hostname

    print(f'Running on machine "{hostname}"')

    hostname_root_dict = {
        'ubuntu20': '/mnt/storage/Cloud/MEGA',  # the name of ionut's machine
        'openlab': '/fs/sdsatumd/ionmodo'
    }
    root_path = os.path.join(hostname_root_dict[hostname], 'TrojAI-data', 'round1-dataset-train', 'models')

    for _id, _label in [(1, 'clean'), (7, 'backdoored')]:
        model_id = f'id-{_id:08d}'
        print(f'----------{model_id} ({_label})----------')
        model_path = os.path.join(root_path, model_id)
        model_confusion_experiment(model_path, model_id, SDNConfig.DenseNet_attach_to_DenseBlocks, device)
    print('script ended')