# ============================ Imports ============================
# Standard library
import argparse
import ast
import glob
import json
import math
import os
import pickle
import random
import sys
import time
import warnings

# 3rd-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft, hilbert
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
from torchvision import transforms

# Silence warnings
warnings.filterwarnings("ignore")

# Local imports: ensure project folders are on sys.path before importing
sys.path.append(os.path.join(os.path.dirname(__file__), ""))
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

# Import only the utilities actually used in this file
from utils.helper_function import (
    set_seed,
    count_model_parameters,
    AverageMeter,
    ProgressMeter,
    sax_tokenizer,
)

# Model and dataset imports
from models.our_models import *
from models.train_utils import *
from utils.dataset_cfg import WESAD, DSADS, DaliaHAR
from data_utils.dataset_builder import *
from thop import profile
from torch.autograd import Function
from math import ceil

# ============================ Argument Parser ============================
def string_to_list(arg):
    try:
        return ast.literal_eval(arg)
    except (ValueError, SyntaxError):
        return arg.split(',')

parser = argparse.ArgumentParser(description='HeteroIrregTS')

parser.add_argument('--modalities', type=string_to_list, default=['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC'], help='List of modalities')
parser.add_argument('--log_comment', default='dsad', type=str)
parser.add_argument('--chkpt_pth', default='DSADS/', type=str)
parser.add_argument('--results_dir', default='DSADS/', type=str)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--cuda_pick', default='cuda:5', type=str)
parser.add_argument('--seed_num', default=2711, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--num_experts', default='4', type=int)
parser.add_argument('--base_factor', default='5', type=int)
parser.add_argument('--modality_drop', default='0.0', type=float, help='Used to define dropout during evaluation.')


args = parser.parse_args()

# ============================ Config Extraction ============================
num_epochs     = args.num_epochs
chkpt_pth      = './saved_chk_dir/' + args.chkpt_pth
log_comment    = args.log_comment
cuda_pick      = args.cuda_pick
seed_num       = args.seed_num
modalities     = args.modalities
batch_size     = args.batch_size
results_dir    = './results_dir/' + args.results_dir
modality_drop     = args.modality_drop

# ============================ Setup ============================
set_seed(seed_num)
device = torch.device(cuda_pick if torch.cuda.is_available() else "cpu")
print(device)

dataset_cfg = DSADS()
modalities = dataset_cfg.modalities
print('Modalities:', modalities)
os.makedirs(chkpt_pth, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

writer = SummaryWriter(comment=log_comment)

# ============================ Dataset ============================
input_length = dataset_cfg.duration * (dataset_cfg.base_sample_rate)

#UPDATE PATH
train_data =  torch.load('/home/payal/HeteroIrregTS/data/DSADS/dsad_diversify_dict_scenario1_src.pt')
test_data =  torch.load('/home/payal/HeteroIrregTS/data/DSADS/dsad_diversify_dict_scenario1_trg.pt')

train_samples_split, val_samples_split, train_labels_split, val_labels_split = train_test_split(
    train_data['samples'],
    train_data['labels'],
    test_size=0.2,
    random_state=seed_num,
    stratify=train_data['labels']
)
# Create dictionaries for train and val splits
train_data = {
    'samples': train_samples_split,
    'labels': train_labels_split
}

val_data = {
    'samples': val_samples_split,
    'labels': val_labels_split
}

train_dataset = DSADSDataset(train_data, modalities, dataset_cfg, args.transform)
val_dataset = DSADSDataset(val_data, modalities, dataset_cfg, args.transform)
eval_dataset = DSADSDataset(test_data, modalities, dataset_cfg, args.transform)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader   = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
eval_dataloader  = DataLoader(eval_dataset, batch_size=32, shuffle=True, num_workers=4)

# ============================ Model Setup ============================
input_dim = sum(dataset_cfg.variates[modality] for modality in modalities)
print('Input dim:', input_dim)

model = CrossAttnTransformerClf(
    cfg=dataset_cfg,
    num_classes=dataset_cfg.num_classes,           
    input_length=input_length,        
    d_model=64,
    nhead=8,
    num_layers=4,
    dropout=0.3,
    verbose=True,
    base_factor=10,
    num_experts=8,
)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
class_loss_criterion = nn.CrossEntropyLoss()
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))

#  ============================ Training Loop ============================
best_val_acc = 0
best_test_acc = 0
best_val_f1 = 0
best_eval_f1 = 0
best_epoch_val = -1
best_epoch_test = -1
best_model_val_state = None
best_model_eval_state = None

test_acc_list = np.zeros(num_epochs)
train_acc_list = np.zeros(num_epochs)
val_acc_list = np.zeros(num_epochs)
warmup_epochs = 10
for epoch in range(num_epochs):
    print('Inside Epoch : ', epoch)

    train_loss, train_acc = train_one_epoch(train_dataloader, model, class_loss_criterion, optimizer, epoch, device, num_epochs, warmup_epochs)
    val_loss, val_acc, val_y_true, val_y_pred = evaluate_one_epoch(val_dataloader, model, class_loss_criterion, epoch, device)
    

    # Compute macro F1
    val_f1 = f1_score(val_y_true, val_y_pred, average='macro')

    writer.add_scalar("Class Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Class Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("F1/val", val_f1, epoch)
    writer.flush()

    # Best validation model by accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch_val = epoch
        best_model_val_state = model.state_dict()

        save_dir = os.path.join(chkpt_pth, log_comment)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model_val_state, os.path.join(save_dir, "best_val_model.pth"))

### load the best validation model and evaluate on test set
model.load_state_dict(best_model_val_state)
model = model.to(device)
model = model.float()
eval_loss, eval_acc, eval_y_true, eval_y_pred = evaluate_one_epoch(eval_dataloader, model, class_loss_criterion, epoch, device, modality_drop)
eval_f1 = f1_score(eval_y_true, eval_y_pred, average='macro')
print(f"Best val F1 score: {best_val_f1:.4f} | Accuracy: {best_val_acc:.4f}.")
print(f"Best eval F1 score: {eval_f1:.4f} | Accuracy: {eval_acc:.4f}.")


# ============================ Save Results ============================
model_stats = {
    'best_val_acc': best_val_acc,
    'best_val_f1': best_val_f1,
    'best_epoch_val': best_epoch_val,
    'best_test_acc': eval_acc,
    'best_eval_f1': eval_f1,
}

filename = os.path.join(results_dir, f"{log_comment}.json")
with open(filename, 'w') as f:
    json.dump(model_stats, f, indent=4)



writer.close()