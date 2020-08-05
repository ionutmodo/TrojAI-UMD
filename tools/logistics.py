import os
import torch
from tools.data import TrojAI
from architectures.SDNs.SDNDenseNet121 import SDNDenseNet121

def _read_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        value = int(f.read())
        return value

def read_model_directory(model_root, num_classes, sdn_type, device):
    """This method needs the full path of a model (.../id-00000001) and returns the model, its label and the dataset with images"""
    dataset_path = os.path.join(model_root, 'example_data')
    ground_truth_path = os.path.join(model_root, 'ground_truth.csv')
    model_path = os.path.join(model_root, 'model.pt')

    print('logistics:read_model_directory - check batch_size!')
    _dataset = TrojAI(folder=dataset_path, batch_size=10, device=device)
    _model_label = (_read_ground_truth(ground_truth_path) == 1)
    _model = torch.load(model_path, map_location=device).eval()
    _model = SDNDenseNet121(_model, input_size=(1, 3, 224, 224), num_classes=num_classes, sdn_type=sdn_type, device=device)

    return _dataset, _model_label, _model
