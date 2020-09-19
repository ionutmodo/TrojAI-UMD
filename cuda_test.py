import numpy as np
import torch
import os
from tools.logistics import get_project_root_path
from tools.network_architectures import load_trojai_model, load_model

device = 'cuda'

path_root = get_project_root_path()
path_cnn = os.path.join(path_root, 'TrojAI-data', 'round1-dataset-train', 'models', 'id-00000001', 'model.pt')
path_sdn = os.path.join(path_root, 'TrojAI-data', 'round1-dataset-train', 'models', 'id-00000001')

data = torch.load(path_cnn, map_location=device)
print(data)

# sdn_model, sdn_params = load_model(path_sdn, 'ics_train100_test0_bs25', device=device, epoch=-1)

# print(sdn_model)
