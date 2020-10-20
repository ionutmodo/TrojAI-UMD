import sys
for folder in ['/umd/architectures', '/umd/tools', '/umd/trojai']:
    if folder not in sys.path:
        sys.path.append(folder)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import os.path
import sys
import pickle
import dill
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
plt.rcParams.update({'figure.autolayout': True})

from bisect import bisect_right
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.autograd import Variable
from pathlib import Path
from random import sample 
from ast import literal_eval
from numpy.linalg import norm

import tools.network_architectures as arcs

# from tools.data import CIFAR10, CIFAR100, TinyImagenet

# from architectures.CNNs.resnet import ResNet50
# from architectures.CNNs.wideresnet import WideResNet
# from robustness.attacker import AttackerModel
# from robustness.datasets import CIFAR

class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')
    #sys.stderr = Logger(log_file, 'err')


class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr,8)
            lrs.append(new_lr)
        return lrs

class InputNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


def load_madry_robust_cifar10(path, device):
    robust_madry_model = ResNet50(num_classes=10).to(device)
    dataset = CIFAR('./data/cifar10')
    robust_madry_model = AttackerModel(robust_madry_model, dataset)
    checkpoint = torch.load('{}/cifar_linf_8.pt'.format(path), pickle_module=dill)
    
    # Makes us able to load models saved with legacy versions
    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'

    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]:v for k,v in sd.items()}
    robust_madry_model.load_state_dict(sd)

    robust_madry_model = robust_madry_model.cnn_model.eval().eval().to(device)
    cifar_normalizer = InputNormalize(torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2023, 0.1994, 0.2010])).to(device)
    robust_madry_model.normalizer = cifar_normalizer

    robust_madry_model.input_size = 32
    robust_madry_model.num_classes = 10

    return robust_madry_model


def load_trades_robust_cifar10(path, device):
    robust_trades_model = WideResNet().to(device)
    #  from https://github.com/yaodongyu/TRADES
    robust_trades_model.load_state_dict(torch.load('{}/model_cifar_wrn.pt'.format(path)))
    robust_trades_model.eval()
    robust_trades_model.input_size = 32
    robust_trades_model.num_classes = 10

    return robust_trades_model

def create_path(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return path

def eval_string(string):
    return literal_eval(string)

def get_random_seed():
    return 1221 # 121 and 1221

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

def set_random_seeds_manual(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_unique_elems(arr_of_arr):
    uniques = set()
    for arr in arr_of_arr:
        uniques = uniques | set(arr)

    return uniques

def single_histogram_with_bin_labels(save_path, save_name, hist_values, bin_label_values, hist_label, bin_label, log_scale=False):
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    nbins = 50
    bins = np.linspace(0., 1., nbins)

    ax1.hist(hist_values, bins=bins, histtype='bar', color='r')
    inds = np.digitize(hist_values, bins)

    labels = []
    for bin_idx in range(nbins):
        bin_indices = inds==bin_idx
        bin_vals = bin_label_values[bin_indices]
        if len(bin_vals) > 0:
            labels.append(np.mean(bin_vals))
        else:
            labels.append(0)

    ax1.set_xlabel(hist_label)

    ax2.scatter(bins, labels, marker='*')

    ax1.set_ylabel('Number of Instances')
    ax2.set_ylabel(bin_label, color='k')

    if log_scale:
        ax1.set_yscale('log', nonposy='clip', basey=10)

    plt.grid(True)
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()

def single_histogram(save_path, save_name, hist_values, label, log_scale=False):
    nbins = 20
    plt.hist(hist_values, bins=nbins)
    plt.axvline(np.mean(hist_values), color='k', linestyle='-', linewidth=3)
    plt.xlabel(label)
    plt.ylabel('Number of Instances')
    if log_scale:
        plt.yscale('log', nonposy='clip', basey=10)

    plt.grid(True)
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()

def overlay_two_histograms(save_path, save_name, hist_first_values, hist_second_values, first_label, second_label, xlabel, title=''):
    plt.hist([hist_first_values, hist_second_values], bins=50, label=[f'{first_label} (black)', f'{second_label} (dotted)'])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(xlabel)
    plt.ylabel('Number of Instances')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()


def get_dataset(dataset, batch_size=128, num_holdout=0):
    if dataset == 'cifar10':
        return CIFAR10(batch_size=batch_size, num_holdout=num_holdout)
    elif dataset == 'cifar100':
        return CIFAR100(batch_size=batch_size, num_holdout=num_holdout)
    elif dataset == 'tinyimagenet':
        return TinyImagenet(batch_size=batch_size, num_holdout=num_holdout)

        

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def reducer_avg(acts, reduced_size=1):
    return F.adaptive_avg_pool2d(acts, reduced_size)

def reducer_max(acts, reduced_size=1):
    return F.adaptive_max_pool2d(acts, reduced_size)

def reducer_std(acts):
    flat = acts.view(acts.size(0), acts.size(1), -1)
    return flat.std(2).view(flat.size(0), flat.size(1), 1, 1)

def reduce_activation(x):
    acts_avg = Flatten()(reducer_avg(x, reduced_size=1))
    acts_max = Flatten()(reducer_max(x, reduced_size=1))
    acts_std = Flatten()(reducer_std(x))
    fwd = torch.cat((acts_avg, acts_max, acts_std), dim=1)
    return fwd


def generate_random_target_labels(true_labels, num_classes):
    target_labels = []
    for label in true_labels:
        cur_label = np.argmax(label)
        target_label = cur_label
        while target_label == cur_label:
            target_label = np.random.randint(0, num_classes)
        
        target_labels.append(target_label)

    return np.array(target_labels)

def shift_labels(true_labels, num_classes, shift_seed):
    target_labels = []
    for label in true_labels:
        cur_label = np.argmax(label)
        target_label = (cur_label + shift_seed) % num_classes
        target_labels.append(target_label)

    return np.array(target_labels)

def safe_list_get(l, idx):
    try:
        return l[idx]
    except IndexError:
        return None

def get_pred_single(model, img):
    prep_img = Variable(torch.from_numpy(img.reshape(1,1,28,28)).float(), requires_grad=True)
    output = model(prep_img)
    return output.max(1, keepdim=True)[1].numpy()[0][0]

def model_exists(models_path, model_name):
    return os.path.isdir(models_path+'/'+model_name)

def file_exists(filename):
    return os.path.isfile(filename) 

def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']

def get_optimizer(model, lr_params, stepsize_params, optimizer='sgd'):
    lr=lr_params[0]
    weight_decay=lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    if optimizer == 'sgd':
        momentum=lr_params[2]
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler

def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device


def normalize(data, normalization_data, normalization_type):
    if normalization_type == 'scale':
        data = np.clip((data - normalization_data['min']) / normalization_data['ptp'], 0, 1)
    elif normalization_type == 'norm':
        data = (data - normalization_data['mean']) / normalization_data['std']
    elif normalization_type == 'ns':
        data = (data - normalization_data['mean']) / normalization_data['std']
        data = np.clip((data - normalization_data['min']) / normalization_data['ptp'], 0, 1)
    elif normalization_type == 'sn':
        data = np.clip((data - normalization_data['min']) / normalization_data['ptp'], 0, 1)
        data = (data - normalization_data['mean']) / normalization_data['std']
    elif normalization_type == 'sigmoid':
        data =  1/(1 + np.exp(-data)) 

    return data

def find_overlap_idx_two_lists(list1, list2):
    both = set(list1).intersection(list2)
    return [list1.index(x) for x in both]
        
def get_loss_criterion():
    return CrossEntropyLoss()

def get_encoder_loss_criterion(train=True):
    if train:
        return BCELoss() #MSELoss()
    else:
        return MSELoss()

def get_list_of_samples_from_list(l, num_samples, sample_size):
    samples = []
    for _ in range(num_samples):
        samples.append(sample(l, sample_size))

    return samples

def get_network_structure(input_size, num_layers, structure_params):
    hidden_sizes = []
    cur_hidden_size = input_size
    for layer in range(num_layers):
        cur_hidden_size = math.ceil(cur_hidden_size * structure_params[layer])
        hidden_sizes.append(cur_hidden_size)

    return hidden_sizes

def get_all_trained_models_info(models_path, use_profiler=False, device='gpu'):
    print('Testing all models in: {}'.format(models_path))

    for model_name in sorted(os.listdir(models_path)):
        try:
            print(model_name)
            model_params = arcs.load_params(models_path, model_name, -1)
            train_time = model_params['total_time']
            num_epochs = model_params['epochs']
            task = model_params['task']
            net_type = model_params['network_type']
            
            top1_test = model_params['test_top1_acc']
            top1_train = model_params['train_top1_acc']
            #top5_test = model_params['test_top5_acc']
            #top5_train = model_params['train_top5_acc']


            print('Top1 Test accuracy: {}'.format(top1_test[-1]))
            #print('Top5 Test accuracy: {}'.format(top5_test[-1]))
            #print('\nTop1 Train accuracy: {}'.format(top1_train[-1]))
            #print('Top5 Train accuracy: {}'.format(top5_train[-1]))

            print('Training time: {}, in {} epochs'.format(train_time, num_epochs))

            if use_profiler:
                model, _ = arcs.load_model(models_path, model_name, epoch=-1)
                model.to(device)
                input_size = model_params['input_size']

                total_ops, total_params = profile(model, input_size, device)
                print("#Ops: %f GOps"%(total_ops/1e9))
                print("#Parameters: %f M"%(total_params/1e6))
        
            print('------------------------')
        except:
            #print('FAIL: {}'.format(model_name))
            continue


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def save_tinyimagenet_classname():
    filename = 'tinyimagenet_classes'
    dataset = get_dataset('tinyimagenet')
    tinyimagenet_classes = {}
    
    for index, name in enumerate(dataset.testset_paths.classes):
        tinyimagenet_classes[index] = name

    with open(filename, 'wb') as f:
        pickle.dump(tinyimagenet_classes, f, pickle.HIGHEST_PROTOCOL)

def get_tinyimagenet_classes(prediction=None):
    filename = 'tinyimagenet_classes'
    with open(filename, 'rb') as f:
        tinyimagenet_classes = pickle.load(f)
    
    if prediction is not None:
        return tinyimagenet_classes[prediction]

    return tinyimagenet_classes

def get_task_num_classes(task):
    if task == 'cifar10':
        return 10
    elif task == 'cifar100':
        return 100
    elif task == 'tinyimagenet':
        return 200


def save_obj(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    if not pickle_exists(filename):
        print('Pickle {} does not exist.'.format(filename))
        return None
        
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def loader_inst_counter(loader, batches_range=None):
    num_instances = 0
    for batch_idx, batch in enumerate(loader):
        if (batches_range is not None) and not (batch_idx >= batches_range[0] and batch_idx < batches_range[1]):
            continue 
        num_instances += len(batch[1])
    
    return num_instances

def loader_batch_counter(loader):
    num_batches = 0
    for _ in loader:
        num_batches += 1
    
    return num_batches


def load_np(file_name):
    if np_exists(file_name) == False:
        print('File: {} does not exist.'.format(file_name))
        return False    
    return np.load('{}.npz'.format(file_name), allow_pickle=True)

def np_exists(file_name):
    return file_exists('{}.npz'.format(file_name)) 

def pickle_exists(file_name):
    return file_exists(file_name)

def loader_get_labels(loader, batches_range=None):
    labels = []
    for batch_idx, batch in enumerate(loader):
        if (batches_range is not None) and not (batch_idx >= batches_range[0] and batch_idx < batches_range[1]):
            continue 
        labels.extend(batch[1])
    
    return np.array(labels)



# soft nearest neighbor distance func
def snnl_dist(x1, x2, temperature):
    euc_dist = norm(x1 - x2)
    p = -((euc_dist * euc_dist)/temperature)
    return np.exp(p)


def safe_subset(arr, indices, idx_range=None):
    if idx_range is None:
        return arr[indices]
    else:
        safe_indices = set(indices) & set(idx_range)
        return arr[list(safe_indices)]



def add_noise_to_behaviors(behaviors, num_behaviors=10, noise_level=0.1):

    if noise_level == 0.0:
        return behaviors
        
    new_behs = np.zeros(behaviors.shape)
    num_groups = int(behaviors.shape[1] / num_behaviors)

    for inst_idx, beh in enumerate(behaviors):
        for group_idx in range(num_groups):
            group_beh = beh[group_idx*num_behaviors: (group_idx+1)*num_behaviors]
            l = np.array([-1]* int(num_behaviors/2) + [1]*int(num_behaviors/2))
            noise = random.uniform(-noise_level,noise_level)
            random.shuffle(l)
            new_behs[inst_idx][group_idx*num_behaviors: (group_idx+1)*num_behaviors] = group_beh + noise*l

    return new_behs
