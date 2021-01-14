"""
    This file is used only once to process Can's data into clean, polygon_4 and polygon_8 as well as
    in instagram filter versions
"""

import io
import numpy
import PIL
import cv2
import skimage
import wand
import numpy as np
import matplotlib.pyplot as plt
import trojai.datagen.instagram_xforms as instagram

data = np.load('../../TrojAI-SyntheticDataset/synthetic_data.npz')

data_clean = data['clean_data']
# data_polygon_4 = data['triggered_data'][:1000, :]
# data_polygon_8 = data['triggered_data'][:2000, :]

image = data_clean[0, :]#.astype(np.uint8)

image = instagram.GothamFilterXForm().filter(wand.image.Image.from_array(image))
image = np.array(image)
if image.shape[2] == 4:
    image = image[:,:,:3]
print(image.shape, image.dtype)
plt.imshow(image.astype(np.uint8))
plt.show()
