import torch
import torch.nn.functional as F
import tools.aux_funcs as af
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNTrojAI import SDNTrojAI
from torchvision.models.googlenet import GoogLeNet, Inception


class SDNGoogLeNet(SDNTrojAI):
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNGoogLeNet, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type == SDNConfig.GoogLeNet, 'SDNGoogLeNet:init - Parameter sdn_type must be SDNConfig.GoogLeNet'
        # for c in self.cnn_model.children():
        #     print(type(c))
        # print('--------------------')

    def forward_w_acts(self, x):
        activations = []
        # N x 3 x 224 x 224
        x = self.cnn_model.conv1(x)
        # N x 64 x 112 x 112
        x = self.cnn_model.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.cnn_model.conv2(x)
        # N x 64 x 56 x 56
        x = self.cnn_model.conv3(x)
        # N x 192 x 56 x 56
        x = self.cnn_model.maxpool2(x)

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        # N x 192 x 28 x 28
        x = self.cnn_model.inception3a(x)
        # N x 256 x 28 x 28
        x = self.cnn_model.inception3b(x)
        # N x 480 x 28 x 28
        x = self.cnn_model.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.cnn_model.inception4a(x)
        # N x 512 x 14 x 14

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        # aux1 = torch.jit.annotate(Optional[Tensor], None)
        # if self.aux1 is not None:
        #     if self.training:
        #         aux1 = self.cnn_model.aux1(x)

        x = self.cnn_model.inception4b(x)
        # N x 512 x 14 x 14
        x = self.cnn_model.inception4c(x)
        # N x 512 x 14 x 14
        x = self.cnn_model.inception4d(x)
        # N x 528 x 14 x 14

        # aux2 is placed here in original architecture
        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        # aux2 = torch.jit.annotate(Optional[Tensor], None)
        # if self.aux2 is not None:
        #     if self.training:
        #         aux2 = self.cnn_model.aux2(x)

        x = self.cnn_model.inception4e(x)
        # N x 832 x 14 x 14
        x = self.cnn_model.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.cnn_model.inception5a(x)
        # N x 832 x 7 x 7
        x = self.cnn_model.inception5b(x)
        # N x 1024 x 7 x 7

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        x = self.cnn_model.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.cnn_model.dropout(x)
        x = self.cnn_model.fc(x)
        # N x 1000 (num_classes)
        return activations, x

    def get_layerwise_model_params(self):
        x = torch.zeros(self.input_size).to(self.device)
        params = []
        # N x 3 x 224 x 224
        x = self.cnn_model.conv1(x)
        # N x 64 x 112 x 112
        x = self.cnn_model.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.cnn_model.conv2(x)
        # N x 64 x 56 x 56
        x = self.cnn_model.conv3(x)
        # N x 192 x 56 x 56
        x = self.cnn_model.maxpool2(x)

        # add SDN output (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        # N x 192 x 28 x 28
        x = self.cnn_model.inception3a(x)
        # N x 256 x 28 x 28
        x = self.cnn_model.inception3b(x)
        # N x 480 x 28 x 28
        x = self.cnn_model.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.cnn_model.inception4a(x)
        # N x 512 x 14 x 14

        # add SDN output (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        # aux1 = torch.jit.annotate(Optional[Tensor], None)
        # if self.aux1 is not None:
        #     if self.training:
        #         aux1 = self.cnn_model.aux1(x)

        x = self.cnn_model.inception4b(x)
        # N x 512 x 14 x 14
        x = self.cnn_model.inception4c(x)
        # N x 512 x 14 x 14
        x = self.cnn_model.inception4d(x)
        # N x 528 x 14 x 14

        # aux2 is placed here in original architecture
        # add SDN output (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        # aux2 = torch.jit.annotate(Optional[Tensor], None)
        # if self.aux2 is not None:
        #     if self.training:
        #         aux2 = self.cnn_model.aux2(x)

        x = self.cnn_model.inception4e(x)
        # N x 832 x 14 x 14
        x = self.cnn_model.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.cnn_model.inception5a(x)
        # N x 832 x 7 x 7
        x = self.cnn_model.inception5b(x)
        # N x 1024 x 7 x 7

        # add SDN output (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        x = self.cnn_model.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.cnn_model.dropout(x)
        x = self.cnn_model.fc(x)
        # N x 1000 (num_classes)
        return params


if __name__ == "__main__":
    device = 'cpu'
    path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round2-train-dataset\id-00000008\model.pt'
    _model = SDNGoogLeNet(torch.load(path, map_location=device),
                          input_size=(1, 3, 224, 224),
                          num_classes=5,
                          sdn_type=SDNConfig.GoogLeNet,
                          device=device)
    _model.eval()
    act, output = _model.forward_w_acts(torch.zeros(1, 3, 224, 224).to(device))
    print(_model.get_layerwise_model_params())
    for a in act:
        print(a.size())
    print(output.size())
