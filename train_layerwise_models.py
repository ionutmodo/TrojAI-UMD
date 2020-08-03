import os
import torch
import tools.aux_funcs as af
import tools.network_architectures as arcs
import tools.model_funcs as mf
from SDNDenseNet121 import SDNDenseNet121
from data import TrojAI
from architectures.SDNs.MLP import LayerwiseClassifiers

def train_wrapper(model, path, model_name, images_folder_name, task, device):
    print(f'training layerwise models: {path}, {model_name}, {task}')
    output_params = model.get_layerwise_model_params()

    mlp_num_layers = 2
    mlp_architecture_param = [2, 2]
    architecture_params = (mlp_num_layers, mlp_architecture_param)

    params = {}
    params['network_type'] = 'layerwise_classifiers'
    params['output_params'] = output_params
    params['architecture_params'] = architecture_params

    # train the mlps
    epochs = 20
    lr_params = (0.001, 0.00001)
    stepsize_params = ([10], [0.1])

    ics = LayerwiseClassifiers(output_params, architecture_params).to(device)
    ics.set_model(model)

    dataset = TrojAI(folder=os.path.join(path, images_folder_name), batch_size=10, device=device)
    optimizer, scheduler = af.get_optimizer(ics, lr_params, stepsize_params, optimizer='adam')
    torch.cuda.empty_cache()
    mf.train_layerwise_classifiers(ics, dataset, epochs, optimizer, scheduler, device)
    arcs.save_model(ics, params, path, '{}_layerwise_classifiers'.format(model_name), epoch=-1)


if __name__ == "__main__":
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))

    device = af.get_pytorch_device()

    task = 'trojai'
    path = '/mnt/storage/Cloud/MEGA/TrojAI/data/trojai-round0-dataset/id-00000005/'
    model_name = 'model.pt'
    images_folder_name = 'example_data'

    model = torch.load(os.path.join(path, model_name)).to(device).eval()
    model = SDNDenseNet121(model, input_size=(1, 3, 224, 224), num_classes=5, sdn_type=0, device=device)
    train_wrapper(model, path, model_name, images_folder_name, task, device)
