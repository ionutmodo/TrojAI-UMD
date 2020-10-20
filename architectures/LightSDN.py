import sys
for folder in ['/umd/architectures', '/umd/tools', '/umd/trojai']:
    if folder not in sys.path:
        sys.path.append(folder)

import torch
import torch.nn as nn
import tools.aux_funcs as af
from SDNConfig import SDNConfig
from SDNDenseNet import SDNDenseNet
from SDNGoogLeNet import SDNGoogLeNet
from SDNInception3 import SDNInception3
from SDNMobileNet2 import SDNMobileNet2
from SDNResNet import SDNResNet
from SDNShuffleNet import SDNShuffleNet
from SDNSqueezeNet import SDNSqueezeNet
from SDNVGG import SDNVGG


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
            SDNConfig.DenseNet_blocks: SDNDenseNet,
            SDNConfig.GoogLeNet: SDNGoogLeNet,
            SDNConfig.Inception3: SDNInception3,
            SDNConfig.MobileNet2: SDNMobileNet2,
            SDNConfig.ResNet: SDNResNet,
            SDNConfig.ShuffleNet: SDNShuffleNet,
            SDNConfig.SqueezeNet: SDNSqueezeNet,
            SDNConfig.VGG: SDNVGG,
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
