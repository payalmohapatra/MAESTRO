import os
import pickle
import torch
import numpy as np
import pandas as pd
import sys
import math
## General pytorch libraries
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from math import sqrt, ceil

# append the data 
sys.path.append('./data/')
sys.path.append('./utils/')
sys.path.append('./models/')

from utils import *
from utils.helper_function import set_seed, count_model_parameters, AverageMeter, ProgressMeter, sax_tokenizer


warmup_epochs = 10
def get_modality_dropout_prob(epoch, num_epochs, warmup_epochs=5, max_dropout=0.5):
    if epoch < warmup_epochs:
        return 0.0
    else:
        # Linear schedule
        progress = min((epoch - warmup_epochs) / (num_epochs - warmup_epochs), 1.0)
        return progress * max_dropout

def train_one_epoch(train_loader, model, class_loss_criterion, optimizer, epoch, device, num_epochs, warmup_epochs):
    loss_meter = AverageMeter('Class Loss', ':.4f')
    acc_meter  = AverageMeter('Class Acc', ':.4f')
    model.train()
    model.zero_grad()
    modality_dropout_prob = get_modality_dropout_prob(epoch, num_epochs, warmup_epochs=warmup_epochs, max_dropout=0.4)

    for i, (x, y) in enumerate(train_loader):
        correct = 0
        x, y = x.to(device).float(), y.to(device)
        class_output, mod_sparse = model(x,modality_dropout_prob)
        # class_output = model(x)
        loss = class_loss_criterion(class_output, y)

        _, predicted = torch.max(class_output.data, 1)
        correct += predicted.eq(y).sum().item()
        acc = correct / x.size(0)

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress = ProgressMeter(len(train_loader), [loss_meter, acc_meter], prefix=f"Epoch: [{epoch}]")
        if (i % 50 == 0) or (i == len(train_loader) - 1):
            progress.display(i)
            if i == len(train_loader) - 1:
                print('End of Epoch', epoch, 'Class loss is', '%.4f' % loss_meter.avg, '    Training accuracy is ', '%.4f' % acc_meter.avg)
    return loss_meter.avg, acc_meter.avg

def evaluate_one_epoch(val_loader, model, class_loss_criterion, epoch, device, modality_drop=0.0):
    loss_meter = AverageMeter('Class Loss', ':.4f')
    acc_meter  = AverageMeter('Class Acc', ':.4f')
    model.eval()
    model.zero_grad()

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device).float(), y.to(device)
            model = model.to(device)
            class_output, _ = model(x, modality_drop)
            # class_output = model(x)
            loss = class_loss_criterion(class_output, y)

            _, predicted = torch.max(class_output.data, 1)
            correct = predicted.eq(y).sum().item()
            acc = correct / x.size(0)

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))

            all_preds.append(predicted.cpu())
            all_labels.append(y.cpu())

            if i == len(val_loader) - 1:
                print('End of Epoch', epoch, 
                      '| Validation Class loss:', f'{loss_meter.avg:.4f}', 
                      '| Validation accuracy:', f'{acc_meter.avg:.4f}')

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return loss_meter.avg, acc_meter.avg, y_true, y_pred
