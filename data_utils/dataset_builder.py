# ============================ Imports ============================
import os
import sys
import glob
import math
import time
import random
import argparse
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
from torchvision import transforms

from scipy.signal import stft, hilbert
from torch.autograd import Function

from thop import profile

from utils.helper_function import set_seed, count_model_parameters, AverageMeter, ProgressMeter, sax_tokenizer
# ============================ Dataset ============================
class HARDataset(Dataset):
    def __init__(self, root_dir, modalities, subjects, cfg, transform='sax', sax_params=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subjects = subjects
        self.modalities = modalities
        self.base_sr = cfg.base_sample_rate
        self.duration = cfg.duration
        self.sampling_rates = cfg.sampling_rates
        self.file_paths = []
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 2}

        for subject in subjects:
            subject_dir = os.path.join(root_dir, subject)
            if os.path.exists(subject_dir):
                self.file_paths.extend(glob.glob(os.path.join(subject_dir, "*.pt")))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])

        def resample_data(mod_data, orig_sr):
            expected_length = int(self.duration * self.base_sr)
            resample_factor = self.base_sr / orig_sr

            if resample_factor < 1:
                step = int(1 / resample_factor)
                resampled = mod_data[::step]
            else:
                if len(mod_data.shape) == 2:
                    time_dim, feature_dim = mod_data.shape
                    mod_data_reshaped = mod_data.permute(1, 0).unsqueeze(0)
                    target_len = int(time_dim * resample_factor)
                    resampled = F.interpolate(mod_data_reshaped, size=target_len, mode='linear', align_corners=False).squeeze(0).permute(1, 0)
                else:
                    raise ValueError(f"Unexpected tensor shape: {mod_data.shape}")

            current_length = resampled.shape[0]
            if current_length > expected_length:
                resampled = resampled[:expected_length]
            elif current_length < expected_length:
                padding_needed = expected_length - current_length
                last_frame = resampled[-1:].repeat(padding_needed, 1)
                resampled = torch.cat([resampled, last_frame], dim=0)

            assert resampled.shape[0] == expected_length
            return resampled

        # Resample and concatenate modalities
        if len(self.modalities) > 1:
            resampled_data = []
            for modality in self.modalities:
                if modality in data:
                    mod_data = data[modality]
                    orig_sr = self.sampling_rates[modality]
                    resampled = resample_data(mod_data, orig_sr)
                    resampled_data.append(resampled)
            x = torch.cat(resampled_data, dim=1)
        else:
            modality = self.modalities[0]
            mod_data = data[modality]
            orig_sr = self.sampling_rates[modality]
            x = resample_data(mod_data, orig_sr)

        # Apply SAX if specified
        if self.transform == 'sax':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_tokenizer(
                    x[:, ch].numpy(),
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]

        y = data['label']
        return x, y


# ============================ Dataset ============================
class DSADSDataset(Dataset):
    def __init__(self, dataset_dict, modalities, cfg, transform=None, sax_params=None):
        self.transform = transform
        self.modalities = modalities
        self.base_sr = cfg.base_sample_rate
        self.duration = cfg.duration
        self.sampling_rates = cfg.sampling_rates
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 1}

        # Load samples and labels
        self.x_data = dataset_dict["samples"]
        self.y_data = dataset_dict["labels"]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Apply SAX if specified
        if self.transform == 'sax':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_tokenizer(
                    x[:, ch],
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]
        return x, y
