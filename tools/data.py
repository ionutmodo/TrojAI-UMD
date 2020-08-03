import torch
import os 

import numpy as np
import aux_funcs as af

from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
import skimage.io


class TrojAI:
    def __init__(self, folder, batch_size=128, num_classes=5, num_holdout=0, opencv_format=True, img_format='png', device='cuda'):
        """opencv_format will be True for rounds 0 and 1 and False for all others"""
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_holdout = num_holdout

        num_workers = 4 if device == 'cpu' else 0

        images, labels = self._get_images(folder, opencv_format, img_format)

        self.dataset = ManualData(images, labels, device)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def _get_images(self, folder, opencv_format, img_format):
        array_images, array_labels = [], []
        for f in os.listdir(folder):
            if f.endswith(img_format):
                image = self._get_single_image(os.path.join(folder, f), opencv_format)
                label = int(f.split('_')[1])
                array_images.append(image)
                array_labels.append(label)

        array_images = np.asarray(array_images)
        array_labels = np.asarray(array_labels)

        return array_images, array_labels

    def _get_single_image(self, path, opencv_format):
        # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
        img = skimage.io.imread(path)
        if opencv_format:
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            img = np.stack((b, g, r), axis=2)

        # perform tensor formatting and normalization explicitly
        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        # img = np.expand_dims(img, 0) # !!! comment this to avoid having dataset of size (500, 1, 3, 224, 224)
        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)
        return img


class CIFAR10:
    def __init__(self, batch_size=128, num_holdout=0):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        if num_holdout > 0 and num_holdout < 1:
            self.num_holdout = int(self.num_train * num_holdout)
        else:
            self.num_holdout = num_holdout

            
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        self.no_aug = transforms.Compose([transforms.ToTensor()])

        self.trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.augmented)

        if self.num_holdout > 0:
            print('Creating holdout set ({})...'.format(self.num_holdout))
            af.set_random_seeds() # Deterministic split
            self.trainset, _ = random_split(self.trainset, (self.num_train-self.num_holdout, self.num_holdout))
            af.set_random_seeds()
            self.no_aug_trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.no_aug)
            _, self.holdout_set = random_split(self.no_aug_trainset, (self.num_train-self.num_holdout, self.num_holdout))
            self.holdout_loader = torch.utils.data.DataLoader(self.holdout_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.holdout_loader_shuffle = torch.utils.data.DataLoader(self.holdout_set, batch_size=batch_size, shuffle=True, num_workers=4)

        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.no_aug)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader_shuffle = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=True, num_workers=4)


class CIFAR100:
    def __init__(self, batch_size=128, num_holdout=0):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000

        if num_holdout > 0 and num_holdout < 1:
            self.num_holdout = int(self.num_train * num_holdout)
        else:
            self.num_holdout = num_holdout

        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        self.no_aug = transforms.Compose([transforms.ToTensor()])

        self.trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.augmented)

        if self.num_holdout > 0:
            print('Creating holdout set ({})...'.format(self.num_holdout))
            af.set_random_seeds()
            self.trainset, _ = random_split(self.trainset, (self.num_train-self.num_holdout, self.num_holdout))
            
            af.set_random_seeds()
            self.no_aug_trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.no_aug)
            _, self.holdout_set = random_split(self.no_aug_trainset, (self.num_train-self.num_holdout, self.num_holdout))
            self.holdout_loader = torch.utils.data.DataLoader(self.holdout_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.holdout_loader_shuffle = torch.utils.data.DataLoader(self.holdout_set, batch_size=batch_size, shuffle=True, num_workers=4)

        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset =  datasets.CIFAR100(root='./data', train=False, download=True, transform=self.no_aug)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader_shuffle = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=True, num_workers=4)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TinyImagenet():
    def __init__(self, batch_size=128, num_holdout=0):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000

        if num_holdout > 0 and num_holdout < 1:
            self.num_holdout = int(self.num_train * num_holdout)
        else:
            self.num_holdout = num_holdout
        
        train_dir = 'data/tiny-imagenet-200/train'
        valid_dir = 'data/tiny-imagenet-200/val/images'
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8), transforms.ToTensor()])
        self.no_aug = transforms.Compose([transforms.ToTensor()])

        self.trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)

        if self.num_holdout > 0:
            print('Creating holdout set ({})...'.format(self.num_holdout))
            af.set_random_seeds()
            self.trainset, _ = random_split(self.trainset, (self.num_train-self.num_holdout, self.num_holdout))
            
            af.set_random_seeds()
            self.no_aug_trainset =  datasets.ImageFolder(train_dir, transform=self.no_aug)
            _, self.holdout_set = random_split(self.no_aug_trainset, (self.num_train-self.num_holdout, self.num_holdout))
            self.holdout_loader = torch.utils.data.DataLoader(self.holdout_set, batch_size=batch_size, shuffle=False, num_workers=8)
            self.holdout_loader_shuffle = torch.utils.data.DataLoader(self.holdout_set, batch_size=batch_size, shuffle=True, num_workers=8)

        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=8)

        self.testset =  datasets.ImageFolder(valid_dir, transform=self.no_aug)
        self.testset_paths = ImageFolderWithPaths(valid_dir, transform=self.no_aug)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=8)
        self.test_loader_shuffle = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=True, num_workers=8)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def create_val_folder():
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join('data/tiny-imagenet-200', 'val/images')  # path where validation data is present now
    filename = os.path.join('data/tiny-imagenet-200', 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ManualDataAE(torch.utils.data.Dataset):
    def __init__(self, data, get_indices=False, device='cpu'):
        super(ManualDataAE, self).__init__()

        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.device=device
        self.get_indices = get_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.get_indices:
            return (self.data[idx], self.data[idx], idx)
        else:
            return (self.data[idx], self.data[idx])


class ManualDatasetAE:
    def __init__(self, train_data, batch_size=64, get_indices=False, device='cpu'):
        self.batch_size = batch_size

        if device == 'cpu':
            num_workers = 4
        else:
            num_workers = 0

        self.train_data = ManualDataAE(train_data, get_indices, device)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)


class ManualData(torch.utils.data.Dataset):
    def __init__(self, data, labels, device='cpu'):
        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.device = device
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])

class ManualDataset:
    def __init__(self, train_data, train_labels, test_data=None, test_labels=None, batch_size=64, device='cpu'):
        self.batch_size = batch_size

        if device == 'cpu':
            num_workers = 4
        else:
            num_workers = 0

        if test_data is not None:
            self.train_data = ManualData(train_data, train_labels, device)
            self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            self.test_data = ManualData(test_data, test_labels, device)
            self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            self.data = ManualData(train_data, train_labels, device)
            self.loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            self.train_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)


def split(data, targets, test_ratio=0.1, random_seed=121, normalize=False):
    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=test_ratio, random_state=random_seed)

    if normalize:
        std = np.std(x_train, axis=0)
        mean = np.mean(x_train, axis=0)

        x_train =  (x_train - mean ) / std
        x_test =  (x_test - mean ) / std
    
    return (x_train, x_test, y_train, y_test)


def split_only_X(data, test_ratio=0.1, random_seed=121):
    x_train, x_test = train_test_split(data, test_size=test_ratio, random_state=random_seed)
    return (x_train, x_test)