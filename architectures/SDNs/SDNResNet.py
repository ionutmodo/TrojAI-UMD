import torch
import tools.aux_funcs as af
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNTrojAI import SDNTrojAI
from torchvision.models.resnet import ResNet


class SDNResNet50(SDNTrojAI):
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNResNet50, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type == SDNConfig.ResNet50, 'SDNResNet50:init - Parameter sdn_type must be SDNConfig.ResNet50'
        # for c in self.model.children():
        #     print(type(c))
        # print('--------------------')

    def forward_w_acts(self, x):
        activations = []
        out = x
        for layer in self.cnn_model.children():
            if isinstance(layer, torch.nn.modules.linear.Linear):
                out = torch.flatten(out, 1)

            out = layer(out)

            if isinstance(layer, torch.nn.modules.container.Sequential):
                activations.append(af.reduce_activation(out))
        #         print(out.size())
        # print('--------------------')
        return activations, out

    def get_layerwise_model_params(self):
        x = torch.zeros(self.input_size).to(self.device)
        params = []
        for layer in self.cnn_model.children():
            if isinstance(layer, torch.nn.modules.linear.Linear):
                break # reached FC part, stop

            x = layer(x)

            if isinstance(layer, torch.nn.modules.container.Sequential):
                feature_size = x.size()[-1]
                feature_channels = x.size()[1]
                params.append((self.num_classes, feature_size, feature_channels))
        return params


# if __name__ == "__main__":
#     device = 'cpu'
#     path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round1-holdout-dataset\id-00000000\model.pt'
#     _model = SDNResNet50(torch.load(path, map_location=device),
#                          input_size=(1, 3, 224, 224),
#                          num_classes=5,
#                          sdn_type=SDNConfig.ResNet50,
#                          device=device)
#     _model.eval()
#     act, output = _model.forward_w_acts(torch.zeros(1, 3, 224, 224).to(device))
#     print(_model.get_layerwise_model_params())
