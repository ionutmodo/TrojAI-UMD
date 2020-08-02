import torch
from torchvision.models import inception, densenet, resnet50
from torch.nn.modules import conv, batchnorm, activation, pooling
from architectures.SDNs.SDNConfig import SDNConfig


def SDNParams_DenseNet121(model, input_size, sdn_type, num_classes=5):
    """
        DenseNet121 has all layers inside a Sequential called "features". It is followed by FC layer (not of interest)
        Returns a list containing (num_classes,
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
            Linear layer that we are not interested in attaching any IC
    """
    assert sdn_type in SDNConfig.DenseNet and isinstance(model, densenet.DenseNet), 'Make sure that SDN type is DenseNet'

    if sdn_type == SDNConfig.DenseNet_end_DenseBlock: # attach_IC_at_DenseBlock_end
        """Returns the logits only at the output of each DenseBlock. Leads to a few number of ICs"""
        x = torch.zeros(input_size)  # an input image to forward through the network to get the input size of each IC
        params = []  # will contain tuples (num_classes, current_input_size, input_features_for_current_IC)
        container = list(model.children())[0]  # iterate only through Sequential variable called "features" at index 0
        for layer in container:
            print(type(layer))
            x = layer(x)
            current_size = x.size()[-1] # the feature maps are always squared
            n_channels = x.size()[1] # the number of feature maps
            if isinstance(layer, (pooling.MaxPool2d, densenet._DenseBlock, densenet._Transition)):
                """attach an IC at maxpool (before each DenseBlock), DenseBlocks and Transitions"""
                params.append((num_classes, current_size, n_channels))
        return params
    elif sdn_type == SDNConfig.DenseNet_inside_DenseBlock: # attach_IC_at_each_DenseLayer_in_DenseBlock
        """Returns the logits of each DenseLayer inside any DenseBlock. Leads to many ICs"""
        x = torch.zeros(input_size) # an input image to forward through the network to get the input size of each IC
        params = [] # will contain tuples (num_classes, current_input_size, input_features_for_current_IC)
        container = list(model.children())[0] # iterate only through Sequential variable called "features" at index 0
        for layer in container:
            print(type(layer))
            current_size = x.size()[-1] # the feature maps are always squared
            n_channels = x.size()[1] # the number of feature maps
            if isinstance(layer, (conv.Conv2d, batchnorm.BatchNorm2d, activation.ReLU)):
                """just forward x because we don't attach any IC here (we are at the early hidden layers of network)"""
                x = layer(x)
            elif isinstance(layer, (pooling.MaxPool2d, densenet._Transition)):
                """attach an IC at maxpool and Transition"""
                x = layer(x)
                params.append((num_classes, current_size, n_channels))
            elif isinstance(layer, densenet._DenseBlock):
                """a _DenseBlock will contain many _DenseLayers and we want to attach one IC at each _DenseLayer"""
                # the following code is taken from _DenseBlock-forward method
                features = [x]
                for name, sublayer in layer.items():
                    print(name, sublayer)
                    new_features = sublayer(features)
                    features.append(new_features)
                    sublayer_current_size = new_features.size()[-1] # the feature maps are always squared
                    sublayer_n_channels = new_features.size()[1] # the number of feature maps
                    params.append((num_classes, sublayer_current_size, sublayer_n_channels))
                x = torch.cat(features, 1)
        return params
    raise RuntimeError('Invalid SDN type for DenseNet121SDN')