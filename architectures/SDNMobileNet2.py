import torch
from torch import nn
import tools.aux_funcs as af
from architectures.SDNConfig import SDNConfig
from architectures.SDNTrojAI import SDNTrojAI
from torchvision.models.mobilenet import InvertedResidual


class SDNMobileNet2(SDNTrojAI):
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNMobileNet2, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type == SDNConfig.MobileNet2, 'SDNMobileNet2:init - Parameter sdn_type must be SDNConfig.MobileNet2'
        # for c in self.cnn_model.features.children():
        #     print(type(c))
        # print('--------------------')

    def forward_w_acts(self, x):
        # # add SDN output (IC)
        # activations.append(af.reduce_activation(x))
        activations = []

        # invertedresidual blocks cover indexes 1..17 in features module
        # add SDN ic at #1, #6, #12 and #17
        inverted_residual_indexes_for_sdn_ics = [1, 6, 12, 17]
        for index, layer in enumerate(self.cnn_model.features.children()):
            x = layer(x)
            if isinstance(layer, InvertedResidual):
                if index in inverted_residual_indexes_for_sdn_ics:
                    # add SDN output (IC)
                    activations.append(af.reduce_activation(x))
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.cnn_model.classifier(x)
        return activations, x

    def get_layerwise_model_params(self):
        # feature_size = x.size()[-1]
        # feature_channels = x.size()[1]
        # params.append((self.num_classes, feature_size, feature_channels))

        x = torch.zeros(self.input_size).to(self.device)
        params = []

        # invertedresidual blocks cover indexes 1..17 in features squential module
        # add SDN ic at #1, #6, #12 and #17
        inverted_residual_indexes_for_sdn_ics = [1, 6, 12, 17]
        for index, layer in enumerate(self.cnn_model.features.children()):
            x = layer(x)
            if isinstance(layer, InvertedResidual):
                if index in inverted_residual_indexes_for_sdn_ics:
                    feature_size = x.size()[-1]
                    feature_channels = x.size()[1]
                    params.append((self.num_classes, feature_size, feature_channels))
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.cnn_model.classifier(x)
        return params


if __name__ == "__main__":
    device = 'cpu'
    path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000024\model.pt'
    _model = SDNMobileNet2(torch.load(path, map_location=device),
                           input_size=(1, 3, 224, 224),
                           num_classes=5,
                           sdn_type=SDNConfig.MobileNet2,
                           device=device)
    _model.eval()
    act, output = _model.forward_w_acts(torch.zeros(1, 3, 224, 224).to(device))
    print(_model.get_layerwise_model_params())
    for a in act:
        print(a.size())
    print(output.size())
