import numpy as np
import torch

from tools.network_architectures import load_trojai_model, load_model

device = 'cuda'

path_cnn = f'/mnt/storage/Cloud/MEGA/TrojAI/TrojAI-data/round1-dataset-train/models/id-00000001/model.pt'
path_sdn = f'/mnt/storage/Cloud/MEGA/TrojAI/TrojAI-data/round1-dataset-train/models/id-00000001/'

sdn_model, sdn_params = load_model(path_sdn, 'ics_train100_test0_bs25', device=device, epoch=-1)

print(sdn_model)
