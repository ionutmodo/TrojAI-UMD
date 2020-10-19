import ast
import PIL.Image
import torch
import os, sys
import math
import pandas as pd
import numpy as np
import wand
from tools.settings import TrojAI_input_size
sys.path.insert(0, 'trojai')
import tools.aux_funcs as af
from torchvision import datasets, transforms
from torch.utils.data import sampler, random_split
from sklearn.model_selection import train_test_split
import skimage.io
import trojai.datagen.instagram_xforms as instagram


def _get_single_image(path, opencv_format):
    # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
    img = skimage.io.imread(path)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    if opencv_format:
        img = np.stack((b, g, r), axis=2)
    else:
        img = np.stack((r, g, b), axis=2)

    # perform tensor formatting and normalization explicitly
    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    # img = np.expand_dims(img, 0) # !!! comment this to avoid having dataset of size (500, 1, 3, 224, 224)
    # normalize the image
    img = img - np.min(img)
    img = img / np.max(img)
    return img


def change_color(trigger, color):
    trigger_np = np.asarray(trigger)
    new_trigger = np.copy(trigger_np)
    w, h, c = new_trigger.shape
    for i in range(w):
        for j in range(h):
            if new_trigger[i, j, 3] == 255:
                for k in range(3):
                    if new_trigger[i, j, k] != 0:
                        new_trigger[i, j, k] = color[k]
    return PIL.Image.fromarray(new_trigger)


def create_backdoored_dataset(dir_clean_data,
                              dir_backdoored_data,
                              trigger_type,
                              trigger_name,
                              trigger_color,
                              trigger_size,
                              triggered_classes,
                              trigger_target_class):
    """
    Creates a backdoored dataset given a clean dataset.
    You can choose trigger fraction, classes to be triggered, target class.
    It also saves a csv file in the backdoored root directory giving details about backdoored images.
    :param dir_clean_data: the directory containing clean samples
    :param dir_backdoored_data: the directory where backdoored dataset will be stored
    :param trigger_type: the type of the trigger; can be 'polygon' or 'filter'
    :param trigger_name: 'square' for polygons and ['gotham', 'kelvin', 'lomo', 'nashville', 'toaster'] for filters
    :param trigger_color: the color of the trigger to be set; only used for polygons, ignored for filters
    :param trigger_size: the size of the bounding rectangle of the trigger (the trigger might have a smaller size)
                         only used for polygons, ignored for filters
    :param triggered_classes: the original classes to be backdoored (poisoned)
    :param trigger_target_class: the class in which backdoored images will be misclassified to
    :return: nothing, but saves backdoored images on the disk at location "dir_backdoored_data"
    """
    # assert trigger_type in ['polygon', 'filter'], 'tools.data.create_backdoored_dataset: invalid trigger type'
    # assert trigger_name in ['square', ], 'tools.data.create_backdoored_dataset: invalid trigger type'

    if not os.path.isdir(dir_backdoored_data):
        os.makedirs(dir_backdoored_data)
    df = pd.DataFrame(columns=['filename_clean', 'filename_backdoored', 'original_label', 'final_label', 'triggered', 'config'])
    n = 0
    # create the df which contains default values at first (path to clean data, original label, not triggered and no cfg)
    for f in os.listdir(dir_clean_data):
        if f.endswith('.png'):
            original_label = int(f.split('_')[1])
            basename_clean = os.path.join(dir_clean_data, f)
            basename_backdoored = os.path.join(dir_backdoored_data, f)
            # initially, there are no triggered classes
            df.loc[n] = [basename_clean, basename_backdoored, original_label, original_label, False, 'none']
            n += 1

    # mark the classes to be triggered based on triggered_classes
    for original_label in set(df['original_label']):
        # if the class is marked to be poisoned
        if triggered_classes == 'all' or original_label in triggered_classes:
            # get df indexes of those classes
            mask = df['original_label'] == original_label
            df_indexes = df[mask].index

            # modify the rows for the poisoned classes
            for index in df_indexes:
                df.at[index, 'final_label'] = trigger_target_class # the label of the image will be the target class
                df.at[index, 'triggered'] = True # mark it as triggered
                filename_clean = df.at[index, 'filename_clean'] # full path of clean image
                basename_backdoored = os.path.basename(filename_clean) # create backdoored filename starting from clean filename
                # trigger_type = 'polygon' if np.random.rand() < p_trigger else 'filter'
                config = {'type': trigger_type}
                if trigger_type == 'polygon':
                    basename_backdoored = basename_backdoored.replace('.png', f'_backdoor_triggered_to_{trigger_target_class}.png')

                    # place trigger in the middle of the image (it should be in the middle of the object)
                    new_x = new_y = int(TrojAI_input_size[-1] / 2) - int(trigger_size / 2)
                    config['x'] = new_x
                    config['y'] = new_y
                    config['size'] = trigger_size
                    config['color'] = trigger_color
                elif trigger_type == 'filter':
                    basename_backdoored = basename_backdoored.replace('.png', f'_backdoor_filter_from_{original_label}.png')
                    # config['name'] = np.random.choice(['gotham', 'kelvin', 'lomo', 'nashville'], size=1)[0]
                    config['name'] = trigger_name
                df.at[index, 'filename_backdoored'] = os.path.join(dir_backdoored_data, basename_backdoored)
                df.at[index, 'config'] = str(config)

    # df = df.append(df2)
    df.to_csv(os.path.join(dir_backdoored_data, 'info.csv'), index=False)

    # prepare trigger depending on values of trigger_type and trigger_name
    polygon_trigger = None
    if trigger_type == 'polygon':
        if trigger_name == 'square':
            polygon_trigger = PIL.Image.fromarray(255 * np.ones((224, 224, 4)).astype(np.uint8))
        else:
            polygon_trigger = PIL.Image.open(trigger_name)

        if trigger_color is not None:
            polygon_trigger = change_color(polygon_trigger, trigger_color)
    elif trigger_type == 'filter':
        pass
    else:
        raise RuntimeError('tools.data.create_backdoored_dataset: invalid trigger_type')

    count = 0
    # this last pass uses the dataframe created above to write down the backdoored images on disk effectively
    for _, row in df.iterrows():
        is_triggered = row['triggered']
        if is_triggered: # if the class is not triggered, do not transform any images
            filename_clean = row['filename_clean']
            filename_backdoored = row['filename_backdoored']
            config = row['config']
            image_clean = PIL.Image.open(filename_clean)

            if config == 'none': # save original image with the backdoored name
                image_clean.save(filename_backdoored)
                count += 1
            else:
                config = ast.literal_eval(config)
                if config['type'] == 'polygon':
                    image_trigger = polygon_trigger.copy().resize((config['size'], config['size']))
                    image_clean.paste(image_trigger, (config['x'], config['y']), image_trigger)
                    image_clean.save(filename_backdoored)
                    count += 1
                elif config['type'] == 'filter':
                    filter = None
                    if config['name'] == 'gotham':
                        filter = instagram.GothamFilterXForm()
                    elif config['name'] == 'kelvin':
                        filter = instagram.KelvinFilterXForm()
                    elif config['name'] == 'lomo':
                        filter = instagram.LomoFilterXForm()
                    elif config['name'] == 'nashville':
                        filter = instagram.NashvilleFilterXForm()
                    elif config['name'] == 'toaster':
                        filter = instagram.ToasterXForm()
                    image_filtered = filter.filter(wand.image.Image.from_array(image_clean))
                    image_filtered.save(filename=filename_backdoored)
                    count += 1


class TrojAI:
    def __init__(self, folder, batch_size=128, num_classes=5, test_ratio=0.2, opencv_format=True, img_format='png', device='cuda'):
        """opencv_format will be True for rounds 0 and 1 and False for all others"""
        self.batch_size = batch_size
        self.test_ratio = test_ratio

        num_workers = 2 if device == 'cpu' else 0

        X, y = self._get_images(folder, opencv_format, img_format)

        self.num_classes = len(set(y))

        if test_ratio == 0:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=self.test_ratio)
        # print(f'TrojAI:init - train_ratio={1-test_ratio}, test_ratio={test_ratio}')
        # print(f'X_train: {X_train.shape}')
        # print(f'y_train has {y_train.shape}')
        # print(f'X_test: {X_test.shape}')
        # print(f'y_test has {y_test.shape}')

        self.train_dataset = ManualData(X_train, y_train, device)
        self.test_dataset = ManualData(X_test, y_test, device)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        # print('TrojAI:init - test_loader IS THE SAME AS train_loader (it is used like this just for debugging purposes)')

    def _get_images(self, folder, opencv_format, img_format):
        array_images, array_labels = [], []
        for f in os.listdir(folder):
            if f.endswith(img_format):
                image = _get_single_image(os.path.join(folder, f), opencv_format)
                label = int(f.split('_')[1])
                array_images.append(image)
                array_labels.append(label)

        array_images = np.asarray(array_images)
        array_labels = np.asarray(array_labels)

        return array_images, array_labels


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
            num_workers = 2
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