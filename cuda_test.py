import torch
import torchvision
import os
from tools.logistics import get_project_root_path
from tools.network_architectures import load_params
from architectures.MLP import LayerwiseClassifiers

print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# print('memory_stats')
# print(torch.cuda.memory_stats(device))

# print('memory_summary')
# print(torch.cuda.memory_summary(device))

os.system('nvidia-smi')
os.system('kill -9 ')


path_root = get_project_root_path()
path_sdn = os.path.join(path_root, 'TrojAI-data', 'round1-dataset-train', 'models', 'id-00000001')
path_load = os.path.join(path_root, 'TrojAI-data', 'round1-dataset-train', 'models', 'id-00000001', 'ics_train100_test0_bs25', 'last')

print('loading model params')
model_params = load_params(path_sdn, 'ics_train100_test0_bs25', epoch=-1)

print('creating LayerwiseClassifiers object')
model = LayerwiseClassifiers(model_params['output_params'], model_params['architecture_params'])

print('loading model state')
model_state = torch.load(path_load, map_location=device)

print('loading state to model')
model.load_state_dict(model_state, strict=False)

print('script ended')
