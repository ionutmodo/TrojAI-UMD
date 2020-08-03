import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet
from torch.nn.modules import conv, batchnorm, activation, pooling
from tools import aux_funcs as af
from architectures.SDNs.SDNConfig import SDNConfig


class SDNDenseNet121(nn.Module):
    def __init__(self, model, input_size, num_classes, sdn_type, device):
        super(SDNDenseNet121, self).__init__()
        assert sdn_type == SDNConfig.DenseNet_end_DenseBlock, 'Only DenseNet_end_DenseBlock mode is supported now'
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes
        self.sdn_type = sdn_type
        self.device = device

    def forward(self, x):
        fwd = self.model(x)
        return fwd

    def forward_w_acts(self, x):
        activations = []
        net_features, net_classifier = list(self.model.children())
        out = None
        # the for below works with internal layers of "features"
        for layer in net_features:  # children[0] is the Sequential containing convolutional blocks (called 'features')
            out = layer(x if out is None else out)
            if isinstance(layer, (pooling.MaxPool2d, densenet._DenseBlock, densenet._Transition)):
                """get the logits at this output stage"""
                activation_3x = af.reduce_activation(out)
                activations.append(activation_3x)
                # print(tuple(out.size()))
                # print(tuple(activation_3x.size()))
                # print()
        # the code below is taken from source code of DenseNet.forward
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = net_classifier(out)  # forward pass the very last feature (output of children[0]) through FC layer at the end
        # print(out.size())
        return activations, out

    def get_layerwise_model_params(self):
        """
            DenseNet121 has all layers inside a Sequential called "features". It is followed by FC layer (not of interest)
            This is how DenseNet121 looks like:
                Sequential:
                    <class 'torch.nn.modules.conv.Conv2d'>
                    <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
                    <class 'torch.nn.modules.activation.ReLU'>
                    <class 'torch.nn.modules.pooling.MaxPool2d'>
                    <class 'torchvision.models.densenet._DenseBlock'>
                        contains multiple objects of type torchvision.models.densenet._DenseLayer
                    <class 'torchvision.models.densenet._Transition'>
                    <class 'torchvision.models.densenet._DenseBlock'>
                        contains multiple objects of type torchvision.models.densenet._DenseLayer
                    <class 'torchvision.models.densenet._Transition'>
                    <class 'torchvision.models.densenet._DenseBlock'>
                        contains multiple objects of type torchvision.models.densenet._DenseLayer
                    <class 'torchvision.models.densenet._Transition'>
                    <class 'torchvision.models.densenet._DenseBlock'>
                        contains multiple objects of type torchvision.models.densenet._DenseLayer
                    <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
                Linear layer that we are not interested in attaching any IC here, so we don't use it
        """
        x = torch.zeros(self.input_size).to(self.device)  # an input image to forward through the network to get the input size of each IC
        params = []  # will contain tuples (self.num_classes, current_input_size, input_features_for_current_IC)
        children = list(self.model.children())
        net_features, net_classifier = children[0], children[1]  # iterate only through Sequential variable called "features" at index 0

        if self.sdn_type == SDNConfig.DenseNet_end_DenseBlock:  # attach_IC_at_DenseBlock_end
            # Returns the logits only at the output of each DenseBlock. Leads to a few number of ICs
            for layer in net_features:
                # print(type(layer))
                x = layer(x)
                current_size = x.size()[-1]  # the feature maps are always squared
                n_channels = x.size()[1]  # the number of feature maps
                if isinstance(layer, (pooling.MaxPool2d, densenet._DenseBlock, densenet._Transition)):
                    # attach an IC at maxpool (before each DenseBlock), DenseBlocks and Transitions
                    params.append((self.num_classes, current_size, n_channels))
            return params
        elif self.sdn_type == SDNConfig.DenseNet_inside_DenseBlock:  # attach_IC_at_each_DenseLayer_in_DenseBlock
            # Returns the logits of each DenseLayer inside any DenseBlock. Will attach to many ICs
            for layer in net_features:
                # print(type(layer))
                current_size = x.size()[-1]  # the feature maps are always squared
                n_channels = x.size()[1]  # the number of feature maps
                if isinstance(layer, (conv.Conv2d, batchnorm.BatchNorm2d, activation.ReLU)):
                    # just forward x because we don't attach any IC here (we are at the early hidden layers of network)
                    x = layer(x)
                elif isinstance(layer, (pooling.MaxPool2d, densenet._Transition)):
                    # attach an IC at maxpool and Transition
                    x = layer(x)
                    params.append((self.num_classes, current_size, n_channels))
                elif isinstance(layer, densenet._DenseBlock):
                    # a _DenseBlock will contain many _DenseLayers and we want to attach one IC at each _DenseLayer
                    # the code below is taken from _DenseBlock-forward method
                    features_list = [x]
                    for name, sublayer in layer.items():
                        # print(name, sublayer)
                        new_features = sublayer(features_list)
                        features_list.append(new_features)
                        sublayer_current_size = new_features.size()[-1]  # the feature maps are always squared
                        sublayer_n_channels = new_features.size()[1]  # the number of feature maps
                        params.append((self.num_classes, sublayer_current_size, sublayer_n_channels))
                    x = torch.cat(features_list, 1)
            return params
