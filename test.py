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
from tools.data import _get_single_image

# path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000001\example_data\class_0_example_2.png'
path = r'/fs/sdsatumd/ionmodo/TrojAI/TrojAI-data/round2-train-dataset/id-00000001/example_data/class_0_example_2.png'
image = PIL.Image.open(path)

filters = {
    'gotham': instagram.GothamFilterXForm(),
    'kelvin': instagram.KelvinFilterXForm(),
    'lomo': instagram.LomoFilterXForm(),
    'nashville': instagram.NashvilleFilterXForm()
}
for name, filter in filters.items():
    image_filtered = filter.filter(wand.image.Image.from_array(image))
    # image_filtered.save(filename=rf'C:\Users\Ionut-Vlad Modoranu\Desktop\{name}.png')
    image_filtered.save(filename=rf'/fs/sdsatumd/ionmodo/TrojAI/TrojAI-data/{name}.png')
