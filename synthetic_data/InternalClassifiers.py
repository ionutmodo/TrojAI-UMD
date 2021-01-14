import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

import aux_funcs as af
from bisect import bisect_right

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss, L1Loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler

import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def reducer_avg(acts, reduced_size=1):
    return F.adaptive_avg_pool2d(acts, reduced_size)

def reducer_max(acts, reduced_size=1):
    return F.adaptive_max_pool2d(acts, reduced_size)

def reducer_std(acts):
    flat = acts.view(acts.size(0), acts.size(1), -1)
    return flat.std(2).view(flat.size(0), flat.size(1), 1, 1)

def reduce_activation(x):
    acts_avg = Flatten()(reducer_avg(x, reduced_size=1))
    acts_max = Flatten()(reducer_max(x, reduced_size=1))
    acts_std = Flatten()(reducer_std(x))
    fwd = torch.stack((acts_max, acts_avg, acts_std), dim=2).view(x.shape[0], 3*x.shape[1])
    return fwd

class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr,8)
            lrs.append(new_lr)
        return lrs

def get_network_structure(input_size, structure_params):
    hidden_sizes = []
    cur_num_neurons = input_size
    for expansion_factor in structure_params:
        cur_hidden_size = math.ceil(cur_num_neurons * expansion_factor)
        hidden_sizes.append(cur_hidden_size)
        cur_num_neurons = cur_hidden_size

    return hidden_sizes

def get_optimizer(model, optim_params):
    lr = optim_params['init_lr']
    wd  = optim_params['weight_decay']
    optimizer = optim_params['optim_type']
    milestones = optim_params['reduce_lr_epochs']
    gammas = optim_params['reduce_lr_factors']

    if optimizer == 'sgd':
        momentum = optim_params.get('momentum', 0.9)

        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def train_ics(ics, loaders, optim_params):

    train_loader, test_loader = loaders

    epochs = optim_params['epochs']

    optimizer, scheduler = get_optimizer(ics, optim_params)

    for epoch in range(1, epochs+1):
        ics.train()
        print('Epoch : {}'.format(epoch))

        for batch in train_loader:
            ics_training_step(ics, optimizer, batch)
        
        if test_loader is not None:
            ics_test_accs = ics_test(ics, test_loader)
            print('Layerwise ICs Test Accs: {}\n'.format(ics_test_accs))

        ics_train_accs = ics_test(ics, train_loader)
        print('\nLayerwise ICs Train Accs: {}\n\n'.format(ics_train_accs))

        scheduler.step()


def NLL(predicted, target):
    return -(target * predicted).sum(dim=1).mean()


def ics_training_step(ics, optimizer, batch):
    b_x = batch[0].cuda().float()
    
    soft_labels = batch[1].ndim == 2 

    b_y = batch[1].cuda().float() if soft_labels else batch[1].cuda().long() 

    outputs, _ = ics(b_x) # logits from ICs

    outputs = [nn.functional.log_softmax(output, dim=1) for output in outputs] if soft_labels else outputs
    criterion = NLL if soft_labels else CrossEntropyLoss()

    loss = 0

    # since the losses in each layer is independent (the main model is frozen), this will simultaneously update all aes
    for output in outputs:
        loss += criterion(output, b_y) 

    optimizer.zero_grad()          
    loss.backward()                
    optimizer.step()
    del loss


def ics_test(ics, loader):
    ics.eval()
    num_corrects = [0] * (ics.num_ics + 1) # +1 is the final classifier
    num_samples = 0

    for batch in loader:
        b_x = batch[0].cuda().float()
    
        soft_labels = batch[1].ndim == 2 
        
        b_y = batch[1].max(1)[1].cuda().long() if soft_labels else batch[1].cuda().long() 

        with torch.no_grad():
            outputs, final_output = ics(b_x)

        num_samples += len(b_x)

        for ic_idx, output in enumerate([*outputs, final_output]):
            cur_correct = int((output.data.max(1)[1] == b_y.data).double().sum().cpu().detach().numpy())
            num_corrects[ic_idx] += cur_correct

    accs = [100*(num_correct/num_samples) for num_correct in num_corrects]
    accs = af.Accuracy(accs)

    return accs


class InternalClassifiers(nn.Module):
    def __init__(self, input_sizes, num_classes, structure_params):
        super(InternalClassifiers, self).__init__()

        mlps = []

        for input_size in input_sizes:
            hidden_sizes = af.get_network_structure(3*input_size, structure_params)
            cur_ic = MLP(3*input_size, num_classes, hidden_sizes)
            mlps.append(cur_ic)

        self.mlps = nn.ModuleList(mlps)
        self.num_ics = len(input_sizes)
        self.activation_extractor = None

    def add_activation_extractor(self, act_ex):
        self.activation_extractor = act_ex
        self.activation_extractor.eval()
        self.activation_extractor.model.eval()

    def remove_activation_extractor(self):
        self.activation_extractor = None

    def forward(self, x):

        assert self.activation_extractor is not None, 'Please add a model to the ICs'

        internal_preds = []

        # we forward the input with no grad to avoid modifying the underlying model
        with torch.no_grad():
            acts, output = self.activation_extractor(x)


        reduced_acts = [reduce_activation(act) for act in acts]

        for layer_idx, act in enumerate(reduced_acts):
            cur_pred = self.mlps[layer_idx](act)
            internal_preds.append(cur_pred)

        return internal_preds, output
    
    def eval(self):
        for mlp_idx in range(self.num_ics):
            self.mlps[mlp_idx].eval()
        self.activation_extractor.eval()
        self.activation_extractor.model.eval()
        return self

    def train(self):
        for mlp_idx in range(self.num_ics):
            self.mlps[mlp_idx].train()        
        self.activation_extractor.eval()
        self.activation_extractor.model.eval()
        return self    

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes):
        super(MLP, self).__init__()
        
        
        if len(hidden_sizes) == 0:
            self.layers = nn.Identity()
            self.out = nn.Linear(input_size, num_classes)

        else:
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


class InternalClassifiers_Simple(object):
    def __init__(self, input_sizes, num_classes, model_type, params):
        self.model_type = model_type

        if model_type == 'knn':
            self.models = [KNeighborsClassifier(n_neighbors=params['num_neighbors']) for idx in range(len(input_sizes)+1)] # +1 is the final
        
        elif model_type == 'gbr':
            self.models = [MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=4) for idx in range(len(input_sizes)+1)] # +1 is the final
        
        elif model_type == 'svr':
            self.models = [MultiOutputRegressor(SVR(kernel=params['kernel'], gamma=params['gamma'], C=params['C'], max_iter=10000), n_jobs=4) for idx in range(len(input_sizes)+1)] # +1 is the final

        self.scalers = [QuantileTransformer(output_distribution='uniform') for idx in range(len(input_sizes)+1)]
        # self.scalers = [RobustScaler(quantile_range=(20,80)) for idx in range(len(input_sizes)+1)]


        self.input_sizes = input_sizes
        self.activation_extractor = None
        self.num_classes = num_classes
        self.num_ics = len(input_sizes)

    def add_activation_extractor(self, act_ex):
        self.activation_extractor = act_ex
        self.activation_extractor.eval()
        self.activation_extractor.model.eval()

    def remove_activation_extractor(self):
        self.activation_extractor = None

    # @ignore_warnings(category=ConvergenceWarning)
    def train_models(self, loader):
        
        # collect the activations from the internal layers
        num_samples = af.loader_inst_counter(loader)

        all_acts = [np.zeros((num_samples, input_size*3)) for input_size in self.input_sizes]
        all_acts.append(np.zeros((num_samples, self.num_classes))) # last layer

        all_labels = []

        cur_idx = 0
        for batch in loader:
            b_x = batch[0].cuda().float()

            with torch.no_grad():
                acts, output = self.activation_extractor(b_x)

            acts = [reduce_activation(act).cpu().detach().numpy() for act in acts]
            acts.append(F.softmax(output, dim=1).cpu().detach().numpy())

            for idx, act in enumerate(acts):
                all_acts[idx][cur_idx:(cur_idx+len(b_x))] = act

            soft_labels = batch[1].ndim == 2

            all_labels.append(batch[1].cpu().detach().numpy())

            cur_idx += len(b_x)

        all_labels = np.vstack(all_labels) if soft_labels else np.concatenate(all_labels)

        for idx, model in enumerate(self.models):
            scaled_acts = self.scalers[idx].fit_transform(all_acts[idx])
            model.fit(scaled_acts, all_labels)

    
    def forward(self, x):

        with torch.no_grad():
            acts, output = self.activation_extractor(x)

        acts = [reduce_activation(act).cpu().detach().numpy() for act in acts]
        acts.append(F.softmax(output, dim=1).cpu().detach().numpy())

        preds = []
        for idx, model in enumerate(self.models):
            scaled_acts = self.scalers[idx].transform(acts[idx])
            if self.model_type == 'knn':
                preds.append(torch.from_numpy(model.predict_proba(scaled_acts)).cuda())
            elif self.model_type in ['gbr', 'svr']:   
                preds.append(torch.from_numpy(model.predict(scaled_acts)).cuda())

        return preds[:-1], preds[-1]

    def __call__(self, x):
        return self.forward(x)

    def eval(self):
        return self

    def train(self):
        return self
