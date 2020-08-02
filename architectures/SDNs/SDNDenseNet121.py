import torch
import torch.nn as nn


class SDNDenseNet121(nn.Module):
    def __init__(self, model, input_size):
        super(SDNDenseNet121, self).__init__()