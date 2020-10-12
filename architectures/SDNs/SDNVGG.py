import torch
from torch import nn
import tools.aux_funcs as af
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNTrojAI import SDNTrojAI
from torchvision.models.vgg import VGG


class SDNVGG(SDNTrojAI):
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNVGG, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type == SDNConfig.VGG, 'SDNVGG:init - Parameter sdn_type must be SDNConfig.VGG'

        # keys: the number of layers features module (different for each architecture)
        # values: list containing child indexes where to attach an SDN output
        #         usually, they are applied at first Conv2d layers with 128, 256 and 512 filters,
        #         including the very last Conv2D layer
        self.dict_children_indexes = {
            29: [4, 11, 18, 25], # VGG11BN (exception: add at Conv2d with 64 filters)
            35: [10, 17, 24, 31], # VGG13BN
            44: [10, 17, 27, 40], # VGG16BN
            53: [10, 17, 30, 49] # VGG19BN
        }

    def forward_w_acts(self, x):
        # # add SDN output (IC)
        # activations.append(af.reduce_activation(x))
        activations = []

        n_children = len(list(self.cnn_model.features.children()))
        indexes_list = self.dict_children_indexes[n_children]
        for index, layer in enumerate(self.cnn_model.features.children()):
            x = layer(x)
            if index in indexes_list:
                # add SDN output (IC)
                activations.append(af.reduce_activation(x))
        x = self.cnn_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.cnn_model.classifier(x)
        return activations, x

    def get_layerwise_model_params(self):
        # feature_size = x.size()[-1]
        # feature_channels = x.size()[1]
        # params.append((self.num_classes, feature_size, feature_channels))

        x = torch.zeros(self.input_size).to(self.device)
        params = []

        n_children = len(list(self.cnn_model.features.children()))
        indexes_list = self.dict_children_indexes[n_children]
        for index, layer in enumerate(self.cnn_model.features.children()):
            x = layer(x)
            if index in indexes_list:
                # add SDN output (IC)
                feature_size = x.size()[-1]
                feature_channels = x.size()[1]
                params.append((self.num_classes, feature_size, feature_channels))
        x = self.cnn_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.cnn_model.classifier(x)
        return params


if __name__ == "__main__":
    device = 'cpu'
    path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000087\model.pt' # VGG11BN
    # path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000006\model.pt' # VGG13BN
    # path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000014\model.pt' # VGG16BN
    # path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000043\model.pt' # VGG19BN
    _model = SDNVGG(torch.load(path, map_location=device),
                    input_size=(1, 3, 224, 224),
                    num_classes=5,
                    sdn_type=SDNConfig.VGG,
                    device=device)
    _model.eval()
    act, output = _model.forward_w_acts(torch.zeros(1, 3, 224, 224).to(device))
    print(_model.get_layerwise_model_params())
    for a in act:
        print(a.size())
    print(output.size())
