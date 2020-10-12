import os, sys
import socket

import pandas as pd
import numpy as np
sys.path.insert(0, 'trojai')
import cv2
import torch
from glob import glob

import PIL
import io
import wand.image

from tools import network_architectures as arcs
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNDenseNet import SDNDenseNet
from tools.data import *
from tools.settings import *
import skimage.io
from skimage.transform import resize
from trojai.datagen.experiment import ClassicExperiment
from trojai.datagen.common_label_behaviors import StaticTarget
import trojai.datagen.config as tdc
import trojai.datagen.xform_merge_pipeline as tdx
import trojai.datagen.instagram_xforms as tinstx
import trojai.datagen.merge_interface as td_merge
from trojai.datagen.image_entity import GenericImageEntity
from trojai.datagen.insert_merges import InsertAtRandomLocation, InsertAtLocation


def test_SDN_for_TrojAI():
    os.chdir('/mnt/storage/Cloud/MEGA/TrojAI/data/trojai-round0-dataset')
    id_dense = 5
    id_resnet = 3
    id_inception = 2
    device = 'cuda'

    for i in [id_dense, id_resnet, id_inception]:
        model_name = f'id-{i:08d}/model.pt'
        model = torch.load(model_name).to(device).eval()

        trojai_model = SDNDenseNet(model,
                                   TrojAI_input_size,
                                   TrojAI_num_classes,
                                   SDNConfig.DenseNet_layers,
                                   device)
        acts, out = trojai_model.forward_w_acts(torch.zeros(TrojAI_input_size).to(device))
        print(len(acts) + 1)
        print(trojai_model.get_layerwise_model_params())
        break


def test_TrojAI_dataset():
    dataset = TrojAI(folder='/mnt/storage/Cloud/MEGA/TrojAI-data/trojai-round0-dataset/id-00000000/example_data',
                     test_ratio=0.2,
                     batch_size=10)
    # print(dataset)
    # for batch in dataset.train_loader:
    #     print(batch[0].size(), batch[1].size())
    #     break


def test_trained_sdn_trojai():
    device = 'cpu'
    model_id = 1
    model_path = f'/mnt/storage/Cloud/MEGA/TrojAI-data/round1-dataset-train/models/id-{model_id:08d}'
    model_name = 'model_layerwise_classifiers'

    model, params = arcs.load_model(model_path, model_name, epoch=-1)

    dataset = TrojAI(folder=f'/mnt/storage/Cloud/MEGA/TrojAI-data/round1-dataset-train/id-{model_id:08d}/example_data', batch_size=10)


def test_backdoor_dataset_creation():
    ################################
    def get_clean_data_csv():
        folder = '/mnt/storage/Cloud/MEGA/TrojAI/TrojAI-data/round1-holdout-dataset/id-00000009/example_data'
        clean_data_csv = pd.DataFrame(columns=['file', 'label'])
        n = 0
        for f in os.listdir(folder):
            if f.endswith('.png'):
                file = os.path.join(folder, f)
                label = int(f.split('_')[1])
                clean_data_csv.loc[n] = [file, label]
                n += 1
        csv_path = os.path.join(folder.replace('example_data', ''), 'clean_data.csv')
        clean_data_csv.to_csv(csv_path)
        return csv_path

    class DummyMerge(td_merge.Merge):
        def do(self, obj1, obj2, random_state_obj):
            pass

    # class Trigger(Entity):
    #     def __init__(self, trigger_path, opencv_format=True):
    #         """
    #         Read a trigger image specified by path.
    #         :param trigger_path: the root directory for triggers (e.g. id-x/triggers)
    #         """
    #         self.trigger = _get_single_image(trigger_path, opencv_format)
    #
    #     def get_data(self):
    #         return self.trigger

    def get_triggers_from_dir(triggers_root_folder, opencv_format=True):
        print('Triggers:init - opencv_format = True (don\'t forget to change it for Round 2)')
        triggers = []
        for f in glob(f'{triggers_root_folder}/*.png'):
            trigger = skimage.io.imread(f)[:, :, :3]
            n = trigger.shape[0]
            new_size = int(n * np.random.randint(low=2, high=26, dtype=np.int) / 100.0) # trigger is between 2% and 25% of image size
            trigger = resize(trigger, (new_size, new_size), anti_aliasing=True)
            triggers.append(GenericImageEntity(trigger.astype(np.uint8)))
        return triggers

    ################################
    dir_root = '/mnt/storage/Cloud/MEGA/TrojAI/TrojAI-data/round1-holdout-dataset/id-00000009/'
    dir_example_data = os.path.join(dir_root, 'example_data')
    dir_triggers = os.path.join(dir_root, 'triggers')
    csv_clean_data = os.path.join(dir_root, 'clean_data.csv')
    csv_experiment_result = os.path.join(dir_root, 'experiment_result.csv')

    import time
    np.random.seed(int(time.time()))
    x, y, side = 85, 85, 55
    n = int(side * np.random.randint(low=5, high=26, dtype=np.int) / 100.0)
    image = PIL.Image.open(os.path.join(dir_example_data, 'class_3_example_1.png'))
    trigger = PIL.Image.open(os.path.join(dir_triggers, 'trigger_9_11.png')).resize((n, n))

    new_x, new_y = x - 1, y - 1
    while not (x <= new_x < x+side-n) and not (y <= new_y < y+side-n):
        new_x = np.random.randint(x, x + side - n)
        new_y = np.random.randint(y, y + side - n)
    image.paste(trigger, (new_x, new_y), trigger)
    image = tinstx.LomoFilterXForm().filter(wand.image.Image.from_array(image))
    print(type(image))
    image.save(filename=os.path.join(dir_root, 'test.png'))
    print(f'n={n}, new_x={new_x}, new_y={new_y}')
    return

    exp = ClassicExperiment(data_root_dir=get_clean_data_csv(), trigger_label_xform=StaticTarget(0))
    exp_result = exp.create_experiment(clean_data_csv=csv_clean_data,
                                       experiment_data_folder=dir_example_data,
                                       mod_filename_filter='*.png',
                                       split_clean_trigger=False,  # default
                                       trigger_frac=0.19,
                                       triggered_classes=[0, 1, 2, 3, 4])
    exp_result.to_csv(csv_experiment_result)

    trigger_cfg = tdc.XFormMergePipelineConfig(
        # setup the list of possible triggers that will be inserted into the CIFAR10 data.
        trigger_list=get_triggers_from_dir(dir_triggers),
        # tell the trigger inserter the probability of sampling each type of trigger specified in the trigger
        # list.  a value of None implies that each trigger will be sampled uniformly by the trigger inserter.
        trigger_sampling_prob=None, # use None for uniform distribution for each trigger in trigger_list
        # List any transforms that will occur to the trigger before it gets inserted.  In this case, we do none.
        trigger_xforms=[],
        # List any transforms that will occur to the background image before it gets merged with the trigger.
        trigger_bg_xforms=[],
        # List how we merge the trigger and the background.  Because we don't insert a point trigger,
        # the merge is just a no-op
        trigger_bg_merge=InsertAtLocation(np.array([[100, 130], [100, 130], [100, 130]])),
        # trigger_bg_merge=InsertAtRandomLocation(
        #     'uniform_random_available',
        #     tdc.ValidInsertLocationsConfig(algorithm='edge_tracing')  # there are some more implicit parameters used (check class documentation)
        # ),
        # A list of any transformations that we should perform after merging the trigger and the background.
        # trigger_bg_merge_xforms=[],
        trigger_bg_merge_xforms=[tinstx.GothamFilterXForm()],
        # trigger_bg_merge_xforms=[tinstx.KelvinFilterXForm()],
        # trigger_bg_merge_xforms=[tinstx.LomoFilterXForm()],
        # Denotes how we merge the trigger with the background.
        merge_type='insert',
        # Specify that all the clean data will be modified.  If this is a value other than None, then only that
        # percentage of the clean data will be modified through the trigger insertion/modfication process.
        per_class_trigger_frac=0.19,
        # Specify which classes will be triggered
        triggered_classes=[0])

    tdx.modify_clean_image_dataset(clean_dataset_rootdir=dir_example_data,
                                   clean_csv_file=get_clean_data_csv(),
                                   output_rootdir=dir_root,
                                   output_subdir='example_data_backdoored',
                                   mod_cfg=trigger_cfg)


if __name__ == '__main__':
    # test_SDN_for_TrojAI()
    # test_TrojAI_dataset()
    # test_trained_sdn_trojai()
    # test_backdoor_dataset_creation()
    print('script ended')

"""
cliff wang: attributions - ECCV2019
rauf izmalov: #clusters of clean dataset is num_classes. When trigger is present, we have +1/-1 class
Xiangyu Zhang: stimulation
Xinqiao Zhang: UCSD DeepInspect NeuralCleanse
"""
