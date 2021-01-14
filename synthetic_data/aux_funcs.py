from pathlib import Path
import math
import torch
import numpy as np
import skimage.io
import os
import random
import pickle
import sys
from datetime import datetime
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def create_path(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return path

def get_network_structure(input_size, structure_params):
    hidden_sizes = []
    cur_num_neurons = input_size
    for expansion_factor in structure_params:
        cur_hidden_size = math.ceil(cur_num_neurons * expansion_factor)
        hidden_sizes.append(cur_hidden_size)
        cur_num_neurons = cur_hidden_size

    return hidden_sizes


class Logger(object):
    def __init__(self, log_file, terminal, mode, terminal_active=True):
        self.terminal = terminal

        self.terminal_active = terminal_active

        self.log = open('{}.{}'.format(log_file, mode), "a")
        self.log.write('\n---------------------------\n{}'.format(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
        self.log.flush()

    def write(self, message):

        if self.terminal_active:
            self.terminal.write(message)
            self.terminal.flush()

        self.log.write(message)
        self.log.flush()

    def flush(self):
        if self.terminal_active:
            self.terminal.flush()

        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, sys.stdout, 'out')
    sys.stderr = Logger(log_file, sys.stderr, 'err')

    print('\n---------------------------\n{}'.format(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
    eprint('\n---------------------------\n{}'.format(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))


def get_balanced_indices(labels, num_classes, sample_per_class=100):
    
    dataset_idx = []

    classes_indices = {}
    for class_idx in range(num_classes):
        classes_indices[class_idx] = np.where(labels == class_idx)[0]

    for class_idx in range(num_classes):
        print('Class: {} - Num Samples: {}'.format(class_idx, len(classes_indices[class_idx])))
        dataset_idx.extend(np.random.choice(classes_indices[class_idx], size=sample_per_class, replace=True))

    return dataset_idx

class ManualData(torch.utils.data.Dataset):
    def __init__(self, data, labels=None, device='cpu'):
        self.data = torch.from_numpy(data)

        if labels is not None:
            self.labels = torch.from_numpy(labels)
        else:
            self.labels = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        
        if self.labels is None:
            return data
        else:
            return (data, self.labels[idx])

def convert_to_pytorch_format(img):
    ndim = len(img.shape)

    if ndim == 3:
        # convert to CHW dimension ordering for pytorch
        img = np.transpose(img, (2, 0, 1))
    elif ndim == 4:
        img = np.transpose(img, (0, 3, 1, 2))

    # normalize the image matching pytorch.transforms.ToTensor()
    img = img / 255.0

    return img

# read_mode -- 'middle_crop' , 'random_crop', 'resize'
def read_images_from_path(datapath, example_img_format, input_size, read_mode, labeled=True, has_alpha=False, for_pytorch=True, read_subset=None):

    fns = [os.path.join(datapath, fn) for fn in os.listdir(datapath) if fn.endswith(example_img_format)]

    fns = fns if read_subset is None else random.sample(fns, read_subset) 

    fns.sort()  # ensure file ordering
    
    images = []

    for fn in fns:
        if isinstance(input_size, tuple): # the output image is randomly size between given sizes (min, max)
            img_size = np.random.randint(input_size[0], input_size[1]+1)
        else:
            img_size = input_size
            
        # read the image (using skimage)
        img = skimage.io.imread(fn)
        img = img.astype(dtype=np.float32)

        h, w, _ = img.shape

        if read_mode == 'resize':
            img = resize(img, (img_size, img_size), preserve_range=True, anti_aliasing=True)

        elif read_mode == 'middle_crop':
            dx = int((w - img_size) / 2)
            dy = int((h - img_size) / 2)
            img = img[dy:dy + img_size, dx:dx + img_size, :]

        elif read_mode == 'random_crop':
            dy = np.random.randint(h - img_size)
            dx = np.random.randint(w - img_size)
            img = img[dy:dy + img_size, dx:dx + img_size, :]

        if for_pytorch:
            img = convert_to_pytorch_format(img)

        images.append(img)

    if not labeled:
        return images

    labels = []
    for fn in fns:
        cur_label = int(os.path.split(fn)[1].split('_')[1])
        labels.append(cur_label)

    return images, labels

def get_img_data(datapath, model_input_size, test_size=0, example_img_format='png'):
    # Inference the example images in data
    images, labels = read_images_from_path(datapath, example_img_format, model_input_size, read_mode='middle_crop', labeled=True, for_pytorch=True)

    images, labels = np.asarray(images).astype(np.float32), np.asarray(labels).astype(int)
    
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42, stratify=labels)
        
        train_data = ManualData(X_train, y_train)
        test_data = ManualData(X_test, y_test)

        return train_data, test_data
    
    else:
        data = ManualData(images, labels)
        return data


def get_loader(data, batch_size=16, shuffle=False, num_workers=4):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class Accuracy(object):
    def __init__(self, accuracies):
        self.accuracies = accuracies

    def __str__(self):
        l = ["{}:{}%".format(idx, int(i)) for idx, i in enumerate(self.accuracies)]
        return ' - '.join(l)


def loader_inst_counter(loader):
    return len(loader.dataset)  

def save_model(model, save_path, model_type):

    if model_type == 'fcn':
        torch.save(model, save_path)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(save_path, model_type):
    if model_type == 'fcn':
        return torch.load(save_path, map_location=torch.device('cuda'))
    else:
        with open(save_path, 'rb') as handle:
            return pickle.load(handle)
