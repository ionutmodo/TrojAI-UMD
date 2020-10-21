import torch
import torch.nn as nn
import tools.aux_funcs as af


class LayerwiseClassifiers(nn.Module):
    def __init__(self, output_params, architecture_params):
        super(LayerwiseClassifiers, self).__init__()

        mlps = []
        # print('LayerwiseClassifiers:init - Total number of ICs is', len(output_params))
        for params in output_params:
            num_classes, _, num_channels = params
            reduced_input_size = 3 * num_channels
            num_layers, structure_params = architecture_params
            hidden_sizes = af.get_network_structure(reduced_input_size, num_layers, structure_params)
            cur_ic = MLP(reduced_input_size, num_classes, hidden_sizes)
            mlps.append(cur_ic)
            # if len(mlps) == 30:
            #     print('LayerwiseClassifiers::init - the number of ICs to train is limited for debugging!')
            #     break

        self.num_output = len(mlps)
        self.mlps = nn.Sequential(*mlps)
        self.model = None
        self.architecture_params = architecture_params
        self.output_params = output_params
        self.num_ics = len(output_params)

    def set_model(self, cnn_model):
        cnn_model = cnn_model.eval()

        for param in cnn_model.parameters():
            param.requires_grad = False

        self.model = cnn_model

    def forward(self, x, include_cnn_out=False, with_grad=False):
        """Set parameter include_cnn_out to True when you perform confusion experiments"""
        assert self.model is not None, 'Set the model first by calling set_model.'
        
        if with_grad:
            fwd, out = self.model.forward_w_acts(x)
        else:
            with torch.no_grad():
                fwd, out = self.model.forward_w_acts(x)

        internal_preds = []
        for layer_act, layerwise_mlp in zip(fwd, self.mlps):
            cur_pred = layerwise_mlp(layer_act)
            internal_preds.append(cur_pred)
        if include_cnn_out:
            internal_preds.append(out)
        return internal_preds

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes):
        super(MLP, self).__init__()
        
        
        self.num_classes = num_classes


        if len(hidden_sizes) > 0:
            layers = []
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(0.5))

            cur_size = hidden_sizes[0]
            for hidden_size in hidden_sizes[1:]:
                layers.append(nn.Linear(cur_size, hidden_size))
                layers.append(nn.ReLU(True))
                layers.append(nn.Dropout(0.5))
                cur_size = hidden_size

            self.layers = nn.Sequential(*layers)
            self.out = nn.Linear(hidden_sizes[-1], num_classes)

        else: # there's no hidden layer, it is only a linear classifier
            self.layers = nn.Sequential() # empty since there's no hidden layer
            self.out = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.out(x)
