import sys
for folder in ['/umd/architectures', '/umd/tools', '/umd/trojai']:
    if folder not in sys.path:
        sys.path.append(folder)

import os
import socket

from tools.data import TrojAI
from SDNDenseNet import *#SDNDenseNet
from SDNGoogLeNet import *#SDNGoogLeNet
from SDNInception3 import *#SDNInception3
from SDNMobileNet2 import *#SDNMobileNet2
from SDNResNet import *#SDNResNet
from SDNShuffleNet import *#SDNShuffleNet
from SDNSqueezeNet import *#SDNSqueezeNet
from SDNVGG import *#SDNVGG


def _read_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        value = int(f.read())
        return value


def read_model_directory(model_root, batch_size, test_ratio, device):
    """This method needs the full path of a model (.../id-00000001) and returns the model, its label and the dataset with images"""
    dataset_path = os.path.join(model_root, 'example_data')
    # ground_truth_path = os.path.join(model_root, 'ground_truth.csv')
    model_path = os.path.join(model_root, 'model.pt')

    # print('logistics:read_model_directory - check batch_size!')
    clean_dataset = TrojAI(folder=dataset_path, test_ratio=test_ratio, batch_size=batch_size, device=device, opencv_format=False)
    # model_label = (_read_ground_truth(ground_truth_path) == 1)
    cnn_model = torch.load(model_path, map_location=device).eval()

    sdn_type = -1
    if isinstance(cnn_model, densenet.DenseNet):
        sdn_type = SDNConfig.DenseNet_blocks
        # ctor = SDNDenseNet
    elif isinstance(cnn_model, GoogLeNet):
        sdn_type = SDNConfig.GoogLeNet
        # ctor = SDNGoogLeNet
    elif isinstance(cnn_model, Inception3):
        sdn_type = SDNConfig.Inception3
        # ctor = SDNInception3
    elif isinstance(cnn_model, MobileNetV2):
        sdn_type = SDNConfig.MobileNet2
        # ctor = SDNMobileNet2
    elif isinstance(cnn_model, ResNet):
        sdn_type = SDNConfig.ResNet
        # ctor = SDNResNet
    elif isinstance(cnn_model, ShuffleNetV2):
        sdn_type = SDNConfig.ShuffleNet
        # ctor = SDNShuffleNet
    elif isinstance(cnn_model, SqueezeNet):
        sdn_type = SDNConfig.SqueezeNet
        # ctor = SDNSqueezeNet
    elif isinstance(cnn_model, VGG):
        sdn_type = SDNConfig.VGG
        # ctor = SDNVGG

    # if ctor is None:
    #     raise RuntimeError('tools.logistics.read_model_directory: invalid model instance')

    dict_type_model = {
        SDNConfig.DenseNet_blocks: SDNDenseNet,
        SDNConfig.GoogLeNet: SDNGoogLeNet,
        SDNConfig.Inception3: SDNInception3,
        SDNConfig.MobileNet2: SDNMobileNet2,
        SDNConfig.ResNet: SDNResNet,
        SDNConfig.ShuffleNet: SDNShuffleNet,
        SDNConfig.SqueezeNet: SDNSqueezeNet,
        SDNConfig.VGG: SDNVGG,
    }

    if sdn_type not in dict_type_model.keys():
        raise RuntimeError('tools.logistics.read_model_directory: invalid model instance')

    # if sdn_type not in dict_type_model.keys():
    #     raise RuntimeError('The SDN type is not yet implemented!')

    cnn_model = dict_type_model[sdn_type](cnn_model,
                                          input_size=(1, 3, 224, 224),
                                          num_classes=clean_dataset.num_classes,
                                          sdn_type=sdn_type,
                                          device=device)
    return clean_dataset, sdn_type, cnn_model


def get_project_root_path():
    """
    Returns the root path of the project on a specific machine.
    To add your custom path, you need to add an entry in the dictionary.
    :return:
    """
    hostname = socket.gethostname()
    if hostname.startswith('openlab'):
        hostname = 'openlab'
    elif hostname.startswith('opensub'):
        hostname = 'opensub'
    hostname_root_dict = { # key = hostname, value = your local root path
        'ubuntu20': '/mnt/storage/Cloud/MEGA/TrojAI',  # the name of ionut's linux machine machine
        'windows10': r'D:\Cloud\MEGA\TrojAI',
        'openlab': '/fs/sdsatumd/ionmodo/TrojAI',
        'opensub': '/fs/sdsatumd/ionmodo/TrojAI',
    }
    print(f'Running on machine "{hostname}"')
    print()
    return hostname_root_dict[hostname]
