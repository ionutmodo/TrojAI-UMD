import os
import numpy as np
import skimage.io
import torch
import torch.nn as nn

import warnings 
warnings.filterwarnings("ignore")

import csv
import aux_funcs as af


class ActivationExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.num_layers = 0

        if layers == 'all':
            for layer in [*self.model.modules()]:
                self.num_layers += int(isinstance(layer, torch.nn.Conv2d))
            print('Collecting activations from all convolutional layers ({} total).'.format(self.num_layers))

        else:
            self.num_layers = len(layers)
            print('Collecting activations from {} layers.'.format(self.num_layers))

        self.activations =  [torch.empty(0).cuda() for _ in range(self.num_layers)]

        self.hook_handles = []

        if layers == 'all':
            layer_id = 0
            for layer in [*self.model.modules()]:
                if isinstance(layer, torch.nn.Conv2d):
                    self.hook_handles.append(layer.register_forward_hook(self.save_outputs_hook(layer_id)))
                    layer_id += 1
        else:
            for layer_id, layer_name in enumerate(layers):
                layer = dict([*self.model.named_modules()])[layer_name]
                self.hook_handles.append(layer.register_forward_hook(self.save_outputs_hook(layer_id)))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self.activations[layer_id] = output
        return fn

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def forward(self, x):
        output = self.model(x)
        return self.activations, output


# get the layer names where we collect the activations from, we find these by printing the models after loading and manually inspecting the models
def get_layer_hook_names(architecture_name):
    if architecture_name == 'resnet101':
        return ["layer1", "layer2", "layer3", "layer4"]

    elif architecture_name == 'vgg16bn':
        return ["features.3", "features.10", "features.20", "features.40"]

    elif architecture_name == 'vgg19bn':
        return ["features.3", "features.10", "features.23", "features.49"]

    elif architecture_name == 'vgg19bn':
        return ["features.3", "features.10", "features.23", "features.49"]

    elif architecture_name == 'resnet50':
        return ["layer1", "layer2", "layer3", "layer4"]

    elif architecture_name == 'resnet18':
        return ["layer1", "layer2", "layer3", "layer4"]

    elif architecture_name == 'vgg11bn':
        return ["features.0", "features.4", "features.11", "features.25"]

    elif architecture_name == 'resnet34':
        return ["layer1", "layer2", "layer3", "layer4"]

    elif architecture_name == 'densenet161':
        return ["features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"]

    elif architecture_name == 'googlenet':
        return ["maxpool1", "maxpool2", "maxpool3", "maxpool4"]

    elif architecture_name == 'wideresnet50':
        return ["layer1", "layer2", "layer3", "layer4"]

    elif architecture_name == 'resnet152':
        return ["layer1", "layer2", "layer3", "layer4"]
        
    elif architecture_name == 'densenet121':
        return ["features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"]

    elif architecture_name == 'shufflenet2_0':
        return ["maxpool", "stage2", "stage3", "stage4"]

    elif architecture_name == 'wideresnet101':
        return ["layer1", "layer2", "layer3", "layer4"]

    elif architecture_name == 'densenet169':
        return ["features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"]

    elif architecture_name == 'shufflenet1_5':
        return ["maxpool", "stage2", "stage3", "stage4"]

    elif architecture_name == 'vgg13bn':
        return ["features.3", "features.10", "features.17", "features.31"]

    elif architecture_name == 'densenet201':
        return ["features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"]

    elif architecture_name == 'squeezenetv1_1':
        return ["features.3", "features.6", "features.9", "features.12"]

    elif architecture_name == 'shufflenet1_0':
        return ["maxpool", "stage2", "stage3", "stage4"]

    elif architecture_name == 'mobilenetv2':
        return ["features.3", "features.6", "features.13", "features.17"]

    elif architecture_name == 'inceptionv3':
        return ["Conv2d_4a_3x3", "Mixed_5d", "Mixed_6e", "Mixed_7c"]



# if predict proba is True, then returns soft labels
def get_preds(model, loader, predict_proba=False, temperature=3):
    model.eval()
    preds = []

    for batch in loader:
        b_x = batch.cuda().float()

        with torch.no_grad():
            logits = model(b_x)

            if predict_proba:
                scaled_logits = logits/temperature
                preds.append(nn.functional.softmax(scaled_logits, dim=1).cpu().detach().numpy())
            else:
                cur_preds = logits.data.max(1)[1].int().cpu().detach().numpy().tolist()
                preds.extend(cur_preds)

    if predict_proba:
        preds = np.vstack(preds)
    else:
        preds = np.array(preds)

    return preds
    