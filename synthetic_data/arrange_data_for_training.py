"""
    This file is used only once to process Can's data into clean, polygon_4 and polygon_8 as well as
    in instagram filter versions
"""

import numpy as np
import trojai.datagen.instagram_xforms as instagram

data = np.load('../../TrojAI-SyntheticDataset/synthetic_data.npz')

data_clean = data['clean_data']
data_polygon_4 = data['triggered_data'][:1000, :]
data_polygon_8 = data['triggered_data'][:2000, :]

print(data_polygon_8.shape)
image = data_clean[0,:]
filter = instagram.GothamFilterXForm()
