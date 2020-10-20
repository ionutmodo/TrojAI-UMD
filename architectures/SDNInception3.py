import sys
for folder in ['/umd/architectures', '/umd/tools', '/umd/trojai']:
    if folder not in sys.path:
        sys.path.append(folder)

import torch
import torch.nn.functional as F
import tools.aux_funcs as af
from SDNConfig import SDNConfig
from SDNTrojAI import SDNTrojAI


class SDNInception3(SDNTrojAI):
    def __init__(self, cnn_model, input_size, num_classes, sdn_type, device):
        super(SDNInception3, self).__init__(cnn_model, input_size, num_classes, sdn_type, device)
        assert sdn_type == SDNConfig.Inception3, 'SDNInception3:init - Parameter sdn_type must be SDNConfig.Inception3'
        # for c in self.cnn_model.children():
        #     print(type(c))
        # print('--------------------')

    def forward_w_acts(self, x):
        activations = []
        # N x 3 x 299 x 299
        x = self.cnn_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.cnn_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.cnn_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.cnn_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.cnn_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        # N x 192 x 35 x 35
        x = self.cnn_model.Mixed_5b(x)  # Inception A
        # N x 256 x 35 x 35
        x = self.cnn_model.Mixed_5c(x)  # Inception A
        # N x 288 x 35 x 35
        x = self.cnn_model.Mixed_5d(x)  # Inception A
        # N x 288 x 35 x 35
        x = self.cnn_model.Mixed_6a(x)  # Inception B

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6b(x)  # Inception C
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6c(x)  # Inception C
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6d(x)  # Inception C
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6e(x)  # Inception C
        # N x 768 x 17 x 17
        # aux_defined = self.cnn_model.training and self.cnn_model.aux_logits
        # if aux_defined:
        #     aux = self.cnn_model.AuxLogits(x)
        # else:
        #     aux = None
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_7a(x)  # Inception D

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        # N x 1280 x 8 x 8
        x = self.cnn_model.Mixed_7b(x)  # Inception E
        # N x 2048 x 8 x 8
        x = self.cnn_model.Mixed_7c(x)  # Inception E
        # N x 2048 x 8 x 8

        # add SDN output (IC)
        activations.append(af.reduce_activation(x))

        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.cnn_model.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.cnn_model.fc(x)
        # N x 1000 (num_classes)
        return activations, x

    def get_layerwise_model_params(self):
        x = torch.zeros(self.input_size).to(self.device)
        params = []
        # N x 3 x 299 x 299
        x = self.cnn_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.cnn_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.cnn_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.cnn_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.cnn_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # add SDN output params (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        # N x 192 x 35 x 35
        x = self.cnn_model.Mixed_5b(x)  # Inception A
        # N x 256 x 35 x 35
        x = self.cnn_model.Mixed_5c(x)  # Inception A
        # N x 288 x 35 x 35
        x = self.cnn_model.Mixed_5d(x)  # Inception A
        # N x 288 x 35 x 35
        x = self.cnn_model.Mixed_6a(x)  # Inception B

        # add SDN output params (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6b(x)  # Inception C
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6c(x)  # Inception C
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6d(x)  # Inception C
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_6e(x)  # Inception C
        # N x 768 x 17 x 17
        aux_defined = self.cnn_model.training and self.cnn_model.aux_logits
        if aux_defined:
            aux = self.cnn_model.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.cnn_model.Mixed_7a(x)  # Inception D

        # add SDN output params (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        # N x 1280 x 8 x 8
        x = self.cnn_model.Mixed_7b(x)  # Inception E
        # N x 2048 x 8 x 8
        x = self.cnn_model.Mixed_7c(x)  # Inception E
        # N x 2048 x 8 x 8

        # add SDN output params (IC)
        feature_size = x.size()[-1]
        feature_channels = x.size()[1]
        params.append((self.num_classes, feature_size, feature_channels))

        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.cnn_model.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.cnn_model.fc(x)
        # N x 1000 (num_classes)
        return params


# if __name__ == "__main__":
#     device = 'cpu'
#     path = r'D:\Cloud\MEGA\TrojAI\TrojAI-data\round1-holdout-dataset\id-00000001\model.pt'
#     _model = SDNInception3(torch.load(path, map_location=device),
#                            input_size=(1, 3, 224, 224),
#                            num_classes=5,
#                            sdn_type=SDNConfig.Inception3,
#                            device=device)
#     _model.eval()
#     act, output = _model.forward_w_acts(torch.zeros(1, 3, 224, 224).to(device))
#     print(_model.get_layerwise_model_params())
