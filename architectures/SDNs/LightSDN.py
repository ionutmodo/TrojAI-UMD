import numpy as np
import torch
import torch.nn as nn
import tools.aux_funcs as af
from architectures.SDNs.SDNConfig import SDNConfig
from architectures.SDNs.SDNDenseNet import SDNDenseNet
from architectures.SDNs.SDNResNet import SDNResNet
from architectures.SDNs.SDNInception3 import SDNInception3


class LightSDN(nn.Module):
    def __init__(self, path_model_cnn, path_model_ics, sdn_type, num_classes, device):
        """
        :param path_model_cnn: full path on disk to cnn model file
        :param path_model_ics: full path on disk to ICs file (SVMs, LogisticRegression etc.)
        :param sdn_type: the SDN type to load
        :param num_classes: number of classes
        :param device: cuda or cpu
        """
        super(LightSDN, self).__init__()

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

    def forward(self, x, include_cnn_out=False):
        activations, out = self.model_cnn.forward_w_acts(x)
        ic_predictions = []
        for act, ic in zip(activations, self.model_ics):
            pred_probas = ic.predict_proba(act.cpu())
            ic_predictions.append(torch.tensor(pred_probas))

        output = ic_predictions
        # concatenate internal activations with CNN output
        if include_cnn_out:
            output += [torch.nn.functional.softmax(out.cpu(), dim=1)]
        # del activations, out
        return output
