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

# Local imports: ensure project folders are on sys.path before importing
sys.path.append(os.path.join(os.path.dirname(__file__), ""))
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "data_utils"))
# Import only the utilities actually used in this file
from utils.helper_function import (
    set_seed,
    count_model_parameters,
    AverageMeter,
    ProgressMeter,
    sax_tokenizer,
)

# Model and dataset imports (explicit where possible)
from models.our_models import *
from models.train_utils import *
from utils.dataset_cfg import WESAD, DSADS, DaliaHAR, MIMIC
from data_utils.mimic_utils import get_dataloader

# Silence warnings
warnings.filterwarnings("ignore")


# ============================ Argument Parser ============================
def string_to_list(arg):
    try:
        return ast.literal_eval(arg)
    except (ValueError, SyntaxError):
        return arg.split(',')

parser = argparse.ArgumentParser(description='HeteroIrregTS')

parser.add_argument('--modalities', type=string_to_list, default=['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC'], help='List of modalities')
parser.add_argument('--log_comment', default='mimic_ours_v2', type=str)
parser.add_argument('--chkpt_pth', default='MIMIC/', type=str)
parser.add_argument('--results_dir', default='MIMICy/', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--cuda_pick', default='cuda:5', type=str)
parser.add_argument('--seed_num', default=2711, type=int)
parser.add_argument('--num_experts', default='4', type=int)
parser.add_argument('--base_factor', default='5', type=int)
parser.add_argument('--modality_drop', default='0.0', type=float, help='Used to define dropout during evaluation.')

# parser.add_argument('--model_name', default='ExpertCNNClf', type=str) 


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
num_experts     = args.num_experts
base_factor         = args.base_factor
modality_drop     = args.modality_drop
# model_name = args.model_name

# ============================ Setup ============================
set_seed(seed_num)
device = torch.device(cuda_pick if torch.cuda.is_available() else "cpu")
print(device)

dataset_cfg = MIMIC()
modalities = dataset_cfg.modalities
print('Modalities:', modalities)
os.makedirs(chkpt_pth, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

writer = SummaryWriter(comment=log_comment)

# ============================ Dataset ============================
# get dataloader for icd9 classification task 7
# train_dataloader, valid_dataloader, test_dataloader = get_dataloader(
#     -1, imputed_path='./data/im.pk', batch_size=batch_size, num_workers=4, transform='sax')


train_dataloader, val_dataloader, eval_dataloader = get_dataloader(
    -1, imputed_path='/home/payal/HeteroIrregTS/data/im.pk', batch_size=batch_size, num_workers=4, transform='sax') #UPDATE PATH
dataset_cfg.num_classes = 6

# ============================ Model Setup ============================
input_dim = sum(dataset_cfg.variates[modality] for modality in modalities)
print('Input dim:', input_dim)


# class CrossAttnTransformerClf(nn.Module):
#     def __init__(self, cfg, num_classes, input_length=256, d_model=64, nhead=8, num_layers_per_modal=2, num_layers=2, dropout=0.1, verbose=True, base_factor=10, num_experts=4):
#         super().__init__()
#         self.modalities = cfg.modalities
#         self.variates = cfg.variates
#         self.num_modalities = len(self.modalities)
#         self.input_length = input_length
#         self.verbose = verbose
#         self.d_model = d_model
#         self.base_factor = base_factor
#         self.num_experts = num_experts

#         # Dynamically create input projection layers
#         self.input_projections = nn.ModuleDict({
#             modality: nn.Linear(self.variates[modality], d_model)
#             for modality in self.modalities
#         })

#         # Positional encoder shared across modalities
#         self.pos_encoder = ModalityPositionalEncoder(
#             d_model=d_model,
#             max_len=input_length,
#             num_modalities=self.num_modalities
#         )
        
#         self.temporal_pos_encoder = TemporalPositionalEncoder(
#             d_model=d_model,
#             max_len=input_length
#         )

#         # Per-modality Informer layers
#         self.per_modal_informers = nn.ModuleDict({
#             modality: nn.ModuleList([
#                 InformerEncoderLayer(
#                     d_model=d_model,
#                     n_heads=nhead,
#                     d_ff=d_model * 4,
#                     dropout=dropout,
#                     factor=base_factor,
#                 ) for _ in range(num_layers_per_modal)
#             ]) for modality in self.modalities
#         })

#         # Final fusion Informer with sparse MoE
#         self.informer_encoder = nn.ModuleList([
#             InformerEncoderLayerWithMoE(
#                 d_model=d_model,
#                 n_heads=nhead,
#                 d_ff=d_model * 4,
#                 dropout=dropout,
#                 factor=base_factor,
#                 num_experts=num_experts,
#                 k=1
#             ) for _ in range(num_layers)
#         ])

#         # Final classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, num_classes)
#         )

#         self.factor_gate = nn.Sequential(
#             nn.Linear(self.num_modalities, self.num_modalities),
#             nn.ReLU(),
#             nn.Linear(self.num_modalities, self.num_modalities),
#             nn.Sigmoid()
#       )

#     def forward(self, x, modality_dropout_prob=0.2, training=True):
#         """
#         x: [B, T, total_features]
#         Assumes modality-wise features are concatenated in order defined by self.modalities
#         """

#         projected_modalities = []
#         start_idx = 0

#         x, modality_mask = modality_dropout(x, self.modalities, self.variates, dropout_prob=modality_dropout_prob, training=training
#         )
#         dynamic_factor = self.factor_gate(modality_mask) * self.base_factor
#         for idx, modality in enumerate(self.modalities):
#             num_vars = self.variates[modality]
#             x_m = x[:, :, start_idx:start_idx + num_vars]
#             start_idx += num_vars
#             # Projection
#             x_m = self.input_projections[modality](x_m)
#             factor = ceil(dynamic_factor[idx] * modality_mask[idx] + 1e-3 * (1 - modality_mask[idx]))

#             # Add modality + temporal position encoding
#             x_m = self.temporal_pos_encoder(x_m)

#             # Pass through per-modality Informer layers
#             for layer in self.per_modal_informers[modality]:
#                 x_m = layer(x_m, factor=factor) ## uses this dynamic factor
            
#             # After per-modal Add modality + temporal position encoding Again
#             x_m = self.pos_encoder(x_m, modality_id=idx)
            
#             projected_modalities.append(x_m)
            

#         # Concatenate across modalities
#         x_cat = torch.cat(projected_modalities, dim=1)  # [B, T_total, d_model]

#         # Final Informer encoder with MoE
#         for layer in self.informer_encoder:
#             x_cat = layer(x_cat, self.base_factor) ##uses fixed factor

#         # Global average pooling
#         x_pooled = torch.mean(x_cat, dim=1)

#         return self.classifier(x_pooled), dynamic_factor


input_length = 24
model = CrossAttnTransformerClf(
    cfg=dataset_cfg,
    num_classes=dataset_cfg.num_classes,            # <-- set your number of classes here
    input_length=input_length,         # <-- time length per modality
    d_model=16,
    nhead=8,
    num_layers=1,
    dropout=0.1,
    verbose=True,
    base_factor=args.base_factor,
    num_experts=args.num_experts
)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
class_loss_criterion = nn.CrossEntropyLoss()
lambda_sparse = 1e-5 ## light sparsity --> increase for aggressive sparsity
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