import torch

import tools.aux_funcs as af
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNDenseNet import SDNDenseNet
from architectures.SDNs.SDNResNet import SDNResNet
from architectures.SDNs.SDNInception3 import SDNInception3


class LightSDN:
    def __init__(self, path_model_cnn, path_model_ics, sdn_type, num_classes, device):
        """
        :param path_model_cnn: full path on disk to cnn model file
        :param path_model_ics: full path on disk to ICs file (SVMs, LogisticRegression etc.)
        :param sdn_type: the SDN type to load
        :param num_classes: number of classes
        :param device: cuda or cpu
        """

        self.dict_type_model = {
            SDNConfig.DenseNet_attach_to_DenseBlocks: SDNDenseNet,
            SDNConfig.DenseNet_attach_to_DenseLayers: SDNDenseNet,
            SDNConfig.ResNet50: SDNResNet,
            SDNConfig.Inception3: SDNInception3
        }

        if sdn_type not in self.dict_type_model.keys():
            raise RuntimeError('The SDN type is not yet implemented!')

        self.model_cnn = self.dict_type_model[sdn_type](torch.load(path_model_cnn, map_location=device).eval(),
                                                        input_size=(1, 3, 224, 224),
                                                        num_classes=num_classes,
                                                        sdn_type=sdn_type,
                                                        device=device)
        self.model_cnn.eval()
        self.model_ics = af.load_obj(path_model_ics)
