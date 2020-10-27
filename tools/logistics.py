import os
import socket
from tools.data import TrojAI
from architectures.SDNDenseNet import *#SDNDenseNet
from architectures.SDNGoogLeNet import *#SDNGoogLeNet
from architectures.SDNInception3 import *#SDNInception3
from architectures.SDNMobileNet2 import *#SDNMobileNet2
from architectures.SDNResNet import *#SDNResNet
from architectures.SDNShuffleNet import *#SDNShuffleNet
from architectures.SDNSqueezeNet import *#SDNSqueezeNet
from architectures.SDNVGG import *#SDNVGG
from torchvision.models import densenet
from torchvision.models.googlenet import GoogLeNet
from torchvision.models.inception import Inception3
from torchvision.models.mobilenet import MobileNetV2
from torchvision.models.resnet import ResNet
from torchvision.models.shufflenetv2 import ShuffleNetV2
from torchvision.models.squeezenet import SqueezeNet
from torchvision.models.vgg import VGG


def _read_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        value = int(f.read())
        return value


def read_model_directory(model_path, dataset_path, batch_size=128, test_ratio=0, device='cpu'):
    # print('logistics:read_model_directory - check batch_size!')
    clean_dataset = TrojAI(folder=dataset_path, test_ratio=test_ratio, batch_size=batch_size, device=device, opencv_format=False)
    # model_label = (_read_ground_truth(ground_truth_path) == 1)
    cnn_model = torch.load(model_path, map_location=device)
    cnn_model = cnn_model.eval()

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
