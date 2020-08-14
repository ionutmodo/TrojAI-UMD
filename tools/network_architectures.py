import sys

import torch
import pickle
import os
import os.path

# from architectures.CNNs.VGG import VGG
# from architectures.CNNs.resnet import ResNet50
# from architectures.CNNs.wideresnet import WideResNet

# from encoder import LayerwiseAutoencoders
from torchvision.models import densenet

from architectures.SDNs.MLP import LayerwiseClassifiers

from architectures.SDNs.SDNDenseNet121 import SDNDenseNet121
from tools.settings import TrojAI_input_size


def get_label_and_confidence_from_logits(logits):
    softmax = torch.nn.functional.softmax(logits.to(logits.device), dim=1)
    label = softmax.max(1)[1].item()
    confidence = torch.max(softmax).item()
    return label, confidence

def load_trojai_model(sdn_path, sdn_name, cnn_name, num_classes, sdn_type, device):
    """
    Loads a TrojAI SDN from disk. The structure should be the following:
    .../id-00000000/ (this is sdn_path)
    |-----ics_model (this is sdn_name directory and we load it using methods in network_architectures)
    |----------last (the model at last epoch)
    |----------params_last (the stats at the last epoch)
    |-----model.pt (the CNN model denoted by cnn_name)
    |-----*other files this method doesn't need*
    """
    sdn_model, sdn_params = load_model(sdn_path, sdn_name, epoch=-1)
    sdn_model = sdn_model.to(device)

    cnn_model = torch.load(os.path.join(sdn_path, cnn_name)).to(device)
    if isinstance(cnn_model, densenet.DenseNet):
        cnn_model = SDNDenseNet121(cnn_model, TrojAI_input_size, num_classes, sdn_type, device)
    else:
        raise RuntimeError(f'SDNTrojAI:load_trojai_model - You are trying to load a SDN model that is not supported ({type(cnn_model)})!')
    sdn_model.set_model(cnn_model)
    return sdn_model

def save_networks(model_name, model_params, models_path):
    cnn_name = model_name+'_cnn'

    print('Saving CNN...')
    model_params['base_model'] = cnn_name
    network_type = model_params['network_type']

    if 'resnet50' in network_type:
        model = ResNet50(num_classes=model_params['num_classes'], feat_scale=model_params['feat_scale'])
    elif 'vgg' in network_type: 
        model = VGG(model_params)
    elif 'wideresnet' in network_type:
        model = WideResNet(num_classes=model_params['num_classes'])

    save_model(model, model_params, models_path, cnn_name, epoch=0)
    
    return cnn_name

def create_vgg16bn(models_path, task, get_params=False):
    print('Creating VGG16BN untrained {} models...'.format(task))

    model_params = get_task_params(task)
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64:
        model_params['fc_layers'] = [2048, 1024]

    model_params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_name = '{}_vgg16bn'.format(task)

    # architecture params
    model_params['network_type'] = 'vgg16'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    model_params['conv_batch_norm'] = True

    get_lr_params(model_params)
    
    if get_params:
        return model_params
    
    return save_networks(model_name, model_params, models_path)


def create_resnet50(models_path, task, get_params=False):
    print('Creating resnet50 untrained {} models...'.format(task))
    model_params = get_task_params(task)

    model_name = '{}_resnet50'.format(task)

    model_params['network_type'] = 'resnet50'
    model_params['num_blocks'] = [3,4,6,3]

    get_lr_params(model_params)
    

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path)


def create_wideresnet(models_path, task, get_params=False):
    print('Creating wideresnet untrained {} models...'.format(task))
    model_params = get_task_params(task)

    model_name = '{}_wideresnet'.format(task)

    model_params['network_type'] = 'wideresnet'

    get_lr_params(model_params)
    

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path)


def create_mobilenet(models_path, task, get_params=False):
    print('Creating MobileNet untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_mobilenet'.format(task)
    
    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path)

def get_task_params(task):
    if task == 'cifar10':
        return cifar10_params()
    elif task == 'cifar100':
        return cifar100_params()
    elif task == 'tinyimagenet':
        return tiny_imagenet_params()

def cifar10_params():
    model_params = {}
    model_params['task'] = 'cifar10'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    model_params['feat_scale'] = 1
    return model_params

def cifar100_params():
    model_params = {}
    model_params['task'] = 'cifar100'
    model_params['input_size'] = 32
    model_params['num_classes'] = 100
    model_params['feat_scale'] = 1

    return model_params

def tiny_imagenet_params():
    model_params = {}
    model_params['task'] = 'tinyimagenet'
    model_params['input_size'] = 64
    model_params['num_classes'] = 200
    model_params['feat_scale'] = 2

    return model_params

def get_lr_params(model_params):
    model_params['momentum'] = 0.9

    network_type = model_params['network_type']

    if 'vgg' in network_type:
        model_params['weight_decay'] = 0.0005

    else:
        model_params['weight_decay'] = 0.0001
    
    model_params['learning_rate'] = 0.1
    model_params['epochs'] = 100
    model_params['milestones'] = [35, 60, 85]
    model_params['gammas'] = [0.1, 0.1, 0.1]


def save_model(model, model_params, models_path, model_name, epoch=-1):
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    network_path = os.path.join(models_path, model_name)

    if not os.path.exists(network_path):
        os.makedirs(network_path)

    # epoch == 0 is the untrained network, epoch == -1 is the last
    if epoch == 0:
        path = os.path.join(network_path, 'untrained')
        params_path = os.path.join(network_path, 'parameters_untrained')
    elif epoch == -1:
        path = os.path.join(network_path, 'last')
        params_path = os.path.join(network_path, 'parameters_last')
    else:
        path = os.path.join(network_path, str(epoch))
        params_path = os.path.join(network_path, f'parameters_{epoch}')

    torch.save(model.state_dict(), path)
    print(f'Saved model to {path}')
    sys.stdout.flush()

    if model_params is not None:
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f, pickle.HIGHEST_PROTOCOL)

def load_params(models_path, model_name, epoch=0):
    params_path = models_path + '/' + model_name
    if epoch == 0:
        params_path = params_path + '/parameters_untrained'
    else:
        params_path = params_path + '/parameters_last'

    with open(params_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params

def load_model(models_path, model_name, epoch=0):
    model_params = load_params(models_path, model_name, epoch)
    network_type = model_params['network_type']
        
    if 'resnet50' in network_type:
        model = ResNet50(num_classes=model_params['num_classes'], feat_scale=model_params['feat_scale'])
        model.input_size = model_params['input_size']
    elif 'vgg' in network_type:
        model = VGG(model_params)
    elif 'wideresnet' in network_type:
        model = WideResNet(num_classes=model_params['num_classes'])
        model.input_size = model_params['input_size']
    elif 'layerwise_autoencoder' in network_type:
        model = LayerwiseAutoencoders(model_params['output_params'], model_params['architecture_params'])
    elif 'layerwise_classifier' in network_type:
        model = LayerwiseClassifiers(model_params['output_params'], model_params['architecture_params'])
        
    network_path = models_path + '/' + model_name

    if epoch == 0: # untrained model
        load_path = os.path.join(network_path, 'untrained')
    elif epoch == -1: # last model
        load_path = os.path.join(network_path, 'last')
    else:
        load_path = os.path.join(network_path, str(epoch))

    model.load_state_dict(torch.load(load_path), strict=False)

    return model, model_params
