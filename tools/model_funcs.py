import sys

import time
import numpy as np

from architectures.SDNs.MLP import MLP
import tools.aux_funcs as af
import tools.data as data_module
import torch
from tools import data
from torch.nn import BCELoss

def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top5 = []
    for output_id in range(model.num_output):
        t1 = data.AverageMeter()
        t5 = data.AverageMeter()
        top1.append(t1)
        top5.append(t5)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 5))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))

    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

    return top1_accs, top5_accs

# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    # outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x, include_cnn_out=True)
            abcd = len(output)
            output = [torch.nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return mean_con, std_con

def compute_confusion(model, loader, device='cpu'):
    model.eval()
    confusion_scores = [] # at index i we will have D(x) = sum over all D_i(x)
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            output = model(b_x, include_cnn_out=True)
            output = [torch.nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = get_confusion_scores(output, None, device)
            for index in range(len(b_x)):
                confusion_scores.append(cur_confusion[index].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    return confusion_scores


def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [torch.nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = get_confusion_scores(output, confusion_stats, device)

            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion


def get_confusion_scores(outputs, normalize=None, device='cpu'):
    p = 1
    confusion_scores = torch.zeros(outputs[0].size(0))
    confusion_scores = confusion_scores.to(device)
    # for output, use in zip(outputs, SDNConfig.DenseNet_Mask):
    #     if use:
    for output in outputs:
            cur_disagreement = torch.nn.functional.pairwise_distance(outputs[-1], output, p=p)
            cur_disagreement = cur_disagreement.to(device)
            for instance_id in range(outputs[0].size(0)):
                confusion_scores[instance_id] += cur_disagreement[instance_id]

    if normalize is not None:
        for instance_id in range(outputs[0].size(0)):
            cur_confusion_score = confusion_scores[instance_id]
            cur_confusion_score = cur_confusion_score - normalize[0]  # subtract mean
            cur_confusion_score = cur_confusion_score / normalize[1]  # divide by the standard deviation
            confusion_scores[instance_id] = cur_confusion_score

    return confusion_scores

def get_sdn_stats(layer_correct, layer_wrong, instance_confusion):
    layer_keys = sorted(list(layer_correct.keys()))

    correct_confusion = []
    wrong_confusion = []

    for inst in layer_correct[layer_keys[-1]]:
        correct_confusion.append(instance_confusion[inst])

    for inst in layer_wrong[layer_keys[-1]]:
        wrong_confusion.append(instance_confusion[inst])

    mean_correct_confusion = np.mean(correct_confusion)
    mean_wrong_confusion = np.mean(wrong_confusion)

    print('Confusion of corrects: {}, Confusion of wrongs: {}'.format(mean_correct_confusion, mean_wrong_confusion))

    return correct_confusion, wrong_confusion


def get_cnn_stats(correct, wrong, instance_confidence):
    # print('get cnn stats')

    correct_confidence = []
    wrong_confidence = []

    for inst in correct:
        correct_confidence.append(instance_confidence[inst])
    for inst in wrong:
        wrong_confidence.append(instance_confidence[inst])

    mean_correct_confidence = np.mean(correct_confidence)
    mean_wrong_confidence = np.mean(wrong_confidence)

    print('Confidence of corrects: {}, Confidence of wrongs: {}'.format(mean_correct_confidence, mean_wrong_confidence))
    return correct_confidence, wrong_confidence


def cnn_get_confidence(model, loader, device='cpu'):
    model.eval()
    correct = set()
    wrong = set()
    instance_confidence = {}
    correct_cnt = 0

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = torch.nn.functional.softmax(output, dim=1)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            pred_prob = model_pred[0].to(device)

            is_correct = pred.eq(b_y.view_as(pred))
            correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)

    return correct, wrong, instance_confidence


def cnn_training_step(model, optimizer, data, labels, device='cpu'):
    b_x = data.to(device, dtype=torch.float)   # batch x
    b_y = labels.to(device, dtype=torch.long)   # batch y
    output = model(b_x)            # cnn final output
    criterion = af.get_loss_criterion()
    loss = criterion(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step


    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
    del loss


def cnn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
        print('Epoch: {}/{}'.format(epoch, epochs))
        cur_lr = af.get_lr(optimizer)
        #print('Cur lr: {}'.format(cur_lr))

        start_time = time.time()
        model.train()
        for x, y in data.train_loader:
            cnn_training_step(model, optimizer, x, y, device)
        
        scheduler.step()

        end_time = time.time()
        epoch_time = int(end_time-start_time)

        if hasattr(data, 'test_loader'):
            top1_test, top5_test = cnn_test(model, data.test_loader, device)
            print('Top1 Test accuracy: {}'.format(top1_test))
            print('Top5 Test accuracy: {}'.format(top5_test))
            metrics['test_top1_acc'].append(top1_test)
            metrics['test_top5_acc'].append(top5_test)
        elif hasattr(data, 'train_loader'):
            if (isinstance(model, MLP) and epoch % 10 == 0) or not isinstance(model, MLP):
                top1_train, top5_train = cnn_test(model, data.train_loader, device)
                print('Top1 Train accuracy: {}'.format(top1_train))
                print('Top5 Train accuracy: {}'.format(top5_train))
                metrics['train_top1_acc'].append(top1_train)
                metrics['train_top5_acc'].append(top5_train)
    
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)
        metrics['lrs'].append(cur_lr)

    return metrics
    

def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = data_module.AverageMeter()
    top5 = data_module.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            b_y = batch[1].to(device, dtype=torch.long)

            output = model(b_x)
            if model.num_classes < 5:
                prec1 = data_module.accuracy(output, b_y, topk=(1, ))
            else:
                prec1, prec5 = data_module.accuracy(output, b_y, topk=(1, 5))
                top5.update(prec5[0], b_x.size(0))

            top1.update(prec1[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]

    if model.num_classes < 5:
        top5_acc = 100.0
    else:
        top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc
    
def cnn_test_w_preds(model, loader, device='cpu', ll=False):
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            output = model(b_x)
            if ll:
                model_pred = output.min(1, keepdim=True)
            else:
                model_pred = output.max(1, keepdim=True)
            
            batch_preds = model_pred[1].cpu().detach().numpy()

            for pred in batch_preds:
                preds.append(pred)
    
    return np.array(preds)


def cnn_test_w_details(model, loader, device='cpu'):
    correctly_classified = []
    wrongly_classified = []

    model.eval()
    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device, dtype=torch.float)
            b_y = batch[1].to(device, dtype=torch.long)
            output = model(b_x)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            is_correct = pred.eq(b_y.view_as(pred))

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size

                if cur_correct == 1:
                    correctly_classified.append(cur_instance_id)
                else:
                    wrongly_classified.append(cur_instance_id)
    
    return np.asarray(correctly_classified), np.asarray(wrongly_classified)


################# LAYERWISE CLASSIFIERS TRAINING ############################
def train_layerwise_classifiers(layerwise_classifiers, data, epochs, optimizer, scheduler, device='cpu'):
    for epoch in range(1, epochs+1):
        print('Epoch : {}'.format(epoch))
        for batch in data.train_loader:
            layerwise_classifiers.train()
            layerwise_classifiers_training_step(layerwise_classifiers, optimizer, batch, device)

        if True: # epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            layerwise_ics_test_accs = layerwise_classifiers_test(layerwise_classifiers, data.test_loader, device)
            print('Layerwise ICs Test Accs: {}'.format(layerwise_ics_test_accs))

            layerwise_ics_train_accs = layerwise_classifiers_test(layerwise_classifiers, data.train_loader, device)
            print('Layerwise ICs Train Accs: {}\n'.format(layerwise_ics_train_accs))
            sys.stdout.flush()

        scheduler.step()


def layerwise_classifiers_training_step(layerwise_classifiers, optimizer, batch, device='cpu'):
    b_x = batch[0].to(device, dtype=torch.float)
    b_y = batch[1].to(device, dtype=torch.long)

    outputs = layerwise_classifiers(b_x, include_cnn_out=False)
    criterion = af.get_loss_criterion()

    loss = 0

    # since the losses in each layer is independent (the main model is frozen), this will simultaneously update all aes
    for output in outputs:
        loss += criterion(output, b_y) 

    optimizer.zero_grad()          
    loss.backward()                
    optimizer.step()
    del loss


def layerwise_classifiers_test(layerwise_classifiers, loader, device='cpu'):
    layerwise_classifiers.eval()
    num_corrects = [0] * layerwise_classifiers.num_ics
    num_samples = 0

    for batch in loader:
        b_x, b_y = batch[0].to(device, dtype=torch.float), batch[1].to(device, dtype=torch.long)
        outputs = layerwise_classifiers(b_x, include_cnn_out=False)
        num_samples += len(b_x)

        for ic_idx, output in enumerate(outputs):
            cur_correct = int((output.data.max(1)[1] == b_y.data).float().sum().cpu().detach().numpy())
            num_corrects[ic_idx] += cur_correct

    accs = [round(100*(num_correct/num_samples), 2) for num_correct in num_corrects]
    return accs

######################## ENCODER - DECODER TRAINING ##################################
def train_layerwise_autoencoders(layerwise_autoencoders, data, epochs, optimizer, scheduler, device='cpu'):
    for epoch in range(1, epochs+1):
        print('Epoch : {}'.format(epoch))
        for batch in data.train_loader:
            layerwise_autoencoders.train()
            autoencoder_training_step(layerwise_autoencoders, optimizer, batch, device)

        if epoch == 1 or epoch % 10 == 0:
            test_reconstruction_losses = autoencoder_test_avg(layerwise_autoencoders, data.test_loader, device)
            print('Avg Test Reconstruction Losses: {}\n'.format(test_reconstruction_losses))

            train_reconstruction_losses = autoencoder_test_avg(layerwise_autoencoders, data.train_loader, device)
            print('\nAvg Train Reconstruction Losses: {}\n\n'.format(train_reconstruction_losses))

        scheduler.step()


def autoencoder_test_avg(layerwise_autoencoders, loader, device='cpu'):
    num_instances = 0
    layerwise_autoencoders.eval()
    losses = [0] * layerwise_autoencoders.num_aes

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            outputs, targets = layerwise_autoencoders.forward_w_targets(b_x)

            num_instances += len(b_x)

            for ae_idx, output in enumerate(outputs):
                losses[ae_idx]+= af.get_encoder_loss_criterion(False)(output, targets[ae_idx]).cpu().detach().numpy()

    losses = [loss/num_instances for loss in losses]
    return losses

def autoencoder_get_latents(layerwise_autoencoders, loader, device='cpu'):
    latents = [list()] * layerwise_autoencoders.num_aes
    layerwise_autoencoders.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            all_latents = layerwise_autoencoders.forward_w_latent(b_x)
            for ae_idx, latent in enumerate(all_latents):
                latents[ae_idx].append(latent.cpu().detach().numpy())

    latents = [np.vstack(latent) for latent in latents]

    return latents

def autoencoder_test(layerwise_autoencoders, loader, device='cpu', dist_type = None):
    num_samples = af.loader_inst_counter(loader)

    l2_dists = np.zeros((layerwise_autoencoders.num_aes, num_samples))
    linf_dists = np.zeros((layerwise_autoencoders.num_aes, num_samples))
    l1_dists = np.zeros((layerwise_autoencoders.num_aes, num_samples))
    bce_dists = np.zeros((layerwise_autoencoders.num_aes, num_samples))
    std_dists = np.zeros((layerwise_autoencoders.num_aes, num_samples))


    if dist_type == None:
        dists = ['l1', 'l2', 'linf', 'bce', 'std']
    else:
        dists = [dist_type]

    outer_sample_idx = 0
    layerwise_autoencoders.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            outputs, targets = layerwise_autoencoders.forward_w_targets(b_x)
            for ae_idx, output in enumerate(outputs):
                target = targets[ae_idx]

                cur_sample = outer_sample_idx
                for instance_idx, _ in enumerate(output):

                    if 'bce' in dists:
                        bce = BCELoss()(output[[instance_idx]], target[[instance_idx]])
                        bce = bce.cpu().detach().numpy()
                        bce_dists[ae_idx][cur_sample] = bce

                    delta = output[instance_idx] - target[instance_idx]

                    if 'l2' in dists:

                        l2 = torch.norm(delta, p=2)
                        l2 = l2.cpu().detach().numpy()
                        l2_dists[ae_idx][cur_sample] = l2

                    if 'l1' in dists:
                        l1 = torch.norm(delta, p=1)
                        l1 = l1.cpu().detach().numpy()
                        l1_dists[ae_idx][cur_sample] = l1

                    if 'linf' in dists:
                        linf = torch.norm(delta, p=float("inf"))
                        linf = linf.cpu().detach().numpy()
                        linf_dists[ae_idx][cur_sample] = linf

                    if 'std' in dists:
                        std = torch.std(delta)
                        std = std.cpu().detach().numpy()
                        std_dists[ae_idx][cur_sample] = std

                    cur_sample += 1

            outer_sample_idx += len(b_x)

    dist_dict =  {'bce': bce_dists, 'l2':l2_dists, 'l1': l1_dists, 'linf': linf_dists, 'std': std_dists}
    return dist_dict


def autoencoder_training_step(layerwise_autoencoders, optimizer, batch, device='cpu'):
    b_x = batch[0].to(device, dtype=torch.float)
    outputs, targets = layerwise_autoencoders.forward_w_targets(b_x)

    criterion = af.get_encoder_loss_criterion(True)

    loss = 0

    # since the losses in each layer is independent (the main model is frozen), this will simultaneously update all aes
    for output, target in zip(outputs, targets):
        loss += criterion(output, target) 

    optimizer.zero_grad()          
    loss.backward()                
    optimizer.step()
    del loss
