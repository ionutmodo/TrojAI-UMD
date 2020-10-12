import torch
import torch.nn.functional as F
from torchvision.models import densenet
import tools.aux_funcs as af
from architectures.SDNs.SDNTrojAI import SDNTrojAI
from architectures.SDNs.SDNConfig import SDNConfig


def _forward_w_acts_for_attaching_at_DenseBlocks(x, net_features):
    activations = []
    out = x
    # the for below works with internal layers of "features"
    for layer in net_features:  # children[0] is the Sequential containing convolutional blocks (called 'features')
        out = layer(out)
        # if isinstance(layer, (pooling.MaxPool2d, densenet._DenseBlock, densenet._Transition)):
        if isinstance(layer, densenet._DenseBlock):
            """get the logits at this output stage"""
            activations.append(af.reduce_activation(out))
    return activations, out


def _forward_w_acts_for_attaching_at_DenseLayers(x, net_features):
    activations = []
    out = x
    # Returns the logits of each DenseLayer inside any DenseBlock. Will attach to many ICs
    for layer in net_features:
        # if isinstance(layer, (conv.Conv2d, batchnorm.BatchNorm2d, activation.ReLU)):
        #     # just forward x because we don't attach any IC here (we are at the early hidden layers of network)
        #     out = layer(out)
        # elif isinstance(layer, (pooling.MaxPool2d, densenet._Transition)):
        #     # attach an IC at maxpool and Transition
        #     out = layer(out)
        #     activations.append(af.reduce_activation(out))
        if isinstance(layer, densenet._DenseBlock):
            # a _DenseBlock will contain many _DenseLayers and we want to attach one IC at each _DenseLayer
            # the code below is taken from _DenseBlock-forward method
            features_list = [out]
            for name, sublayer in layer.items():
                new_features = sublayer(features_list)
                features_list.append(new_features)
                activations.append(af.reduce_activation(new_features))
            out = torch.cat(features_list, 1)
        else:
            out = layer(out)
    return activations, out


def _get_layerwise_params_attaching_at_DenseBlocks(x, net_features, num_classes):
    params = [] # will contain tuples (self.num_classes, current_input_size, input_features_for_current_IC)
    # Returns the logits only at the output of each DenseBlock. Leads to a few number of ICs
    for layer in net_features:
        # print(type(layer))
        x = layer(x)
        current_size = x.size()[-1]  # the feature maps are always squared
        n_channels = x.size()[1]  # the number of feature maps
        # if isinstance(layer, (pooling.MaxPool2d, densenet._DenseBlock, densenet._Transition)):
        if isinstance(layer, densenet._DenseBlock):
            # attach an IC at maxpool (before each DenseBlock), DenseBlocks and Transitions
            params.append((num_classes, current_size, n_channels))
    return params


def _get_layerwise_params_attaching_at_DenseLayers(x, net_features, num_classes):
    params = [] # will contain tuples (self.num_classes, current_input_size, input_features_for_current_IC)
    # Returns the logits of each DenseLayer inside any DenseBlock. Will attach to many ICs
    for layer in net_features:
        # print(type(layer))
        # current_size = x.size()[-1]  # the feature maps are always squared
        # n_channels = x.size()[1]  # the number of feature maps
        # if isinstance(layer, (conv.Conv2d, batchnorm.BatchNorm2d, activation.ReLU)):
        #     # just forward x because we don't attach any IC here (we are at the early hidden layers of network)
        #     x = layer(x)
        # elif isinstance(layer, (pooling.MaxPool2d, densenet._Transition)):
        #     # attach an IC at maxpool and Transition
        #     x = layer(x)
        #     params.append((num_classes, current_size, n_channels))
        if isinstance(layer, densenet._DenseBlock):
            # a _DenseBlock will contain many _DenseLayers and we want to attach one IC at each _DenseLayer
            # the code below is taken from _DenseBlock-forward method
            features_list = [x]
            for name, sublayer in layer.items():
                # print(name, sublayer)
                new_features = sublayer(features_list)
                features_list.append(new_features)
                sublayer_current_size = new_features.size()[-1]  # the feature maps are always squared
                sublayer_n_channels = new_features.size()[1]  # the number of feature maps
                params.append((num_classes, sublayer_current_size, sublayer_n_channels))
            x = torch.cat(features_list, 1)
        else: # for conv.Conv2d, batchnorm.BatchNorm2d, activation.ReLU, pooling.MaxPool2d, densenet._Transition
            x = layer(x)
    return params


class SDNDenseNet(SDNTrojAI):
    """
        DenseNet121 has all layers inside a Sequential called "features". It is followed by FC layer (not of interest)
        This is how DenseNet121 looks like:
            features (Sequential):
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
            classifier: linear layer that we are not interested in attaching any IC here, so we don't use it
    """
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNDenseNet, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type in SDNConfig.DenseNet, 'SDNDenseNet121:init - Parameter sdn_type must be in SDNConfig.DenseNet'

    # def forward(self, x): - IT IS IMPLEMENTED IN SDNTrojAI for all models

    def forward_w_acts(self, x):
        net_features, net_classifier = list(self.cnn_model.children())

        if self.sdn_type == SDNConfig.DenseNet_blocks:
            activations, out = _forward_w_acts_for_attaching_at_DenseBlocks(x, net_features)
        elif self.sdn_type == SDNConfig.DenseNet_layers:
            activations, out = _forward_w_acts_for_attaching_at_DenseLayers(x, net_features)
        else:
            raise RuntimeError('SDNDenseNet121:forward_w_acts - invalid value for parameter sdn_type')

        # the code below is taken from source code of DenseNet.forward
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = net_classifier(out)  # forward pass the very last feature (output of children[0]) through FC layer at the end
        return activations, out

    def get_layerwise_model_params(self):
        x = torch.zeros(self.input_size).to(self.device)  # an input image to forward through the network to get the input size of each IC
        net_features = list(self.cnn_model.children())[0] # iterate only through Sequential variable called "features" at index 0

        if self.sdn_type == SDNConfig.DenseNet_blocks:  # attach_IC_at_DenseBlock_end
            return _get_layerwise_params_attaching_at_DenseBlocks(x, net_features, self.num_classes)
        elif self.sdn_type == SDNConfig.DenseNet_layers:  # attach_IC_at_each_DenseLayer_in_DenseBlock
            return _get_layerwise_params_attaching_at_DenseLayers(x, net_features, self.num_classes)
        else:
            raise RuntimeError('get_layerwise_model_params: Invalid sdn_type')
