import torch
import torch.nn as nn
import tools.aux_funcs as af


class LayerwiseClassifiers(nn.Module):
    def __init__(self, output_params, architecture_params):
        super(LayerwiseClassifiers, self).__init__()

        mlps = []
        for params in output_params:
            num_classes, _ , num_channels = params
            reduced_input_size = 3 * num_channels
            num_layers, structure_params = architecture_params
            hidden_sizes = af.get_network_structure(reduced_input_size, num_layers, structure_params)
            cur_ic = MLP(reduced_input_size, num_classes, hidden_sizes)
            mlps.append(cur_ic)

        self.mlps = nn.Sequential(*mlps)
        self.model = None
        self.architecture_params = architecture_params
        self.output_params = output_params
        self.num_ics = len(output_params)

    def set_model(self, model):
        model = model.eval()

        for param in model.parameters():
            param.requires_grad = False

        self.model = model

    def forward(self, x, with_grad=False):
        assert self.model is not None, 'Set the model first by calling set_model.'
        
        if with_grad:
            fwd, _ = self.model.forward_w_acts(x)
        else:
            with torch.no_grad():
                fwd, _ = self.model.forward_w_acts(x)

        internal_preds = []
        for layer_act, layerwise_mlp in zip(fwd, self.mlps):
            cur_pred = layerwise_mlp(layer_act)
            internal_preds.append(cur_pred)

        return internal_preds

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes):
        super(MLP, self).__init__()
        
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
        self.num_classes = num_classes

    def forward(self, x):
        x = self.layers(x)
        return self.out(x)