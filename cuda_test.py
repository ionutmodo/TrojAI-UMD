import numpy as np
import torch
import os
from tools.logistics import get_project_root_path
from tools.network_architectures import load_trojai_model, load_model, load_params
from architectures.SDNs.MLP import LayerwiseClassifiers

device = 'cuda'

path_root = get_project_root_path()
path_sdn = os.path.join(path_root, 'TrojAI-data', 'round1-dataset-train', 'models', 'id-00000001')
path_load = os.path.join(path_root, 'TrojAI-data', 'round1-dataset-train', 'models', 'id-00000001', 'ics_train100_test0_bs25', 'last')

print('loading model params')
model_params = load_params(path_sdn, 'ics_train100_test0_bs25', epoch=-1)

print('creating LayerwiseClassifiers object')
model = LayerwiseClassifiers(model_params['output_params'], model_params['architecture_params'])

print('loading model state')
model_state = torch.load(path_load, map_location=device)

print('loading ')
model.load_state_dict(model_state, strict=False)

print('script ended')
