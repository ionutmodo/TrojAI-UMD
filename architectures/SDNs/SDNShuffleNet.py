import torch
from torch import nn
import tools.aux_funcs as af
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNTrojAI import SDNTrojAI
from torchvision.models.shufflenetv2 import ShuffleNetV2


class SDNShuffleNet(SDNTrojAI):
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNShuffleNet, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type == SDNConfig.ShuffleNet, 'SDNShuffleNet:init - Parameter sdn_type must be SDNConfig.ShuffleNet'

    def forward_w_acts(self, x):
        # # add SDN output (IC)
        # activations.append(af.reduce_activation(x))
        activations = []

        x = self.cnn_model.conv1(x)
        x = self.cnn_model.maxpool(x)
        x = self.cnn_model.stage2(x)

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        x = self.cnn_model.stage3(x)

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        x = self.cnn_model.stage4(x)

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        x = self.cnn_model.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.cnn_model.fc(x)

        return activations, x

    def get_layerwise_model_params(self):
        # feature_size = x.size()[-1]
        # feature_channels = x.size()[1]
        # params.append((self.num_classes, feature_size, feature_channels))

        x = torch.zeros(self.input_size).to(self.device)
        params = []

        x = self.cnn_model.conv1(x)
        x = self.cnn_model.maxpool(x)
        x = self.cnn_model.stage2(x)

        # add SDN output (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        x = self.cnn_model.stage3(x)

        # add SDN output (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        x = self.cnn_model.stage4(x)

        # add SDN output (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        x = self.cnn_model.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.cnn_model.fc(x)

        return params


if __name__ == "__main__":
    device = 'cpu'
    path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000040\model.pt'
    _model = SDNShuffleNet(torch.load(path, map_location=device),
                           input_size=(1, 3, 224, 224),
                           num_classes=5,
                           sdn_type=SDNConfig.ShuffleNet,
                           device=device)
    _model.eval()
    act, output = _model.forward_w_acts(torch.zeros(1, 3, 224, 224).to(device))
    print(_model.get_layerwise_model_params())
    for a in act:
        print(a.size())
    print(output.size())
