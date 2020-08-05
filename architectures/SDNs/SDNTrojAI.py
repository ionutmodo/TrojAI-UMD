import torch.nn as nn
from abc import abstractmethod


class SDNTrojAI(nn.Module):
    def __init__(self, model, input_size, num_classes, sdn_type, device):
        super(SDNTrojAI, self).__init__()
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes
        self.sdn_type = sdn_type
        self.device = device

    def forward(self, x):
        fwd = self.model(x)
        return fwd

    @abstractmethod
    def forward_w_acts(self, x):
        raise NotImplementedError('SDNTrojAI:forward_w_acts - not implemented')

    @abstractmethod
    def get_layerwise_model_params(self):
        raise NotImplementedError('SDNTrojAI:get_layerwise_model_params - not implemented')