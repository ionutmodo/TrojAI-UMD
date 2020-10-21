import torch
import tools.aux_funcs as af
from architectures.SDNConfig import SDNConfig
from architectures.SDNTrojAI import SDNTrojAI
from torchvision.models.squeezenet import Fire


class SDNSqueezeNet(SDNTrojAI):
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNSqueezeNet, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type == SDNConfig.SqueezeNet, 'SDNSqueezeNet:init - Parameter sdn_type must be SDNConfig.SqueezeNet'

    def forward_w_acts(self, x):
        # # add SDN output (IC)
        # activations.append(af.reduce_activation(x))
        activations = []

        # regardless of the network version, count Fire blocks from 1 to 8
        # and place an SDN IC at even indexes (2, 4, 6, 8)
        fire_count = 0
        for layer in self.cnn_model.features.children():
            x = layer(x)
            if isinstance(layer, Fire):
                fire_count += 1
                if fire_count % 2 == 0:
                    # add SDN output (IC)
                    activations.append(af.reduce_activation(x))
        x = self.cnn_model.classifier(x)
        x = torch.flatten(x, 1)
        return activations, x

    def get_layerwise_model_params(self):
        # feature_size = x.size()[-1]
        # feature_channels = x.size()[1]
        # params.append((self.num_classes, feature_size, feature_channels))

        x = torch.zeros(self.input_size).to(self.device)
        params = []

        # regardless of the network version, count Fire blocks from 1 to 8
        # and place an SDN IC at even indexes (2, 4, 6, 8)
        fire_count = 0
        for layer in self.cnn_model.features.children():
            x = layer(x)
            if isinstance(layer, Fire):
                fire_count += 1
                if fire_count % 2 == 0:
                    # add SDN output (IC)
                    feature_size = x.size()[-1]
                    feature_channels = x.size()[1]
                    params.append((self.num_classes, feature_size, feature_channels))
        x = self.cnn_model.classifier(x)
        return params


if __name__ == "__main__":
    device = 'cpu'
    path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000037\model.pt'
    _model = SDNSqueezeNet(torch.load(path, map_location=device),
                           input_size=(1, 3, 224, 224),
                           num_classes=5,
                           sdn_type=SDNConfig.SqueezeNet,
                           device=device)
    _model.eval()
    act, output = _model.forward_w_acts(torch.zeros(1, 3, 224, 224).to(device))
    print(_model.get_layerwise_model_params())
    for a in act:
        print(a.size())
    print(output.size())
