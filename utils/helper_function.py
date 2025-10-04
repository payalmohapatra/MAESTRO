
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import scipy.signal as sig
import matplotlib.pyplot as plt
import IPython.display as ipd
import math
from tqdm import tqdm

#from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
import sklearn
from torch.utils.tensorboard import SummaryWriter
import random
from torch.autograd import Function
from scipy.stats import norm

##################################################################################################
# First things first! Set a seed for reproducibility.
# https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
##################################################################################################
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
##################################################################################################

class ReverseLayerF(Function):
    # Source :  https://torcheeg.readthedocs.io/en/latest/_modules/torcheeg/trainers/domain_adaption/dann.html

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

###################################################################
# ****** Different learning rate schedulers *********************** 
###################################################################
def adjust_learning_rate_cosine_anealing(optimizer, init_lr, epoch, num_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside adjusting cosine lr = ', cur_lr)

def adjust_learning_rate_warmup_time(optimizer, init_lr, epoch, num_epochs, model_size, warmup):
    """Decay the learning rate based on warmup schedule based on time
    Source :: https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer 
    """
    cur_lr = (model_size ** (-0.5) * min((epoch+1) ** (-0.5), (epoch+1) * warmup ** (-1.5))) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside adjusting warmup decay lr = ', cur_lr)

def naive_lr_decay(optimizer, init_lr, epoch, num_epochs):
    """
    Make 3 splits in the num_epochs and just use that to decay the lr 
    """
    if (epoch < np.ceil(num_epochs/4)) :
        cur_lr = init_lr
    elif (epoch < np.ceil(num_epochs/2)) :
        cur_lr = 0.5 * init_lr
    else :
        cur_lr = 0.25 * init_lr    

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside naive decay lr = ', cur_lr)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter():
    def __init__(self, num_batches, meters, prefix=""): 
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

### Pytorch-native preprocessing functions

def normalize_tensor(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std



def preprocess_ppg(signal, fs=64):
    # Ensure we're working with a contiguous array
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    if len(signal) < 10:  # Skip very short signals
        return signal
    
    # Remove baseline wander (high-pass filter at 0.5 Hz)
    # Normalize cutoff frequency by Nyquist frequency (fs/2)
    b, a = sig.butter(3, 0.5/(fs/2), 'high')
    filtered = sig.filtfilt(b, a, signal)
    
    # Band-pass filter to keep cardiac frequency components (0.5-8Hz)
    # Heart rate range: 30-480 BPM = 0.5-8 Hz
    b, a = sig.butter(3, [0.5/(fs/2), 8.0/(fs/2)], 'band')
    filtered = sig.filtfilt(b, a, filtered)
    
    # Normalize
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    return filtered

class PPGTransform:
    def __init__(self, fs=64):
        """
        Initialize PPG transformation with specified sampling frequency
        
        Parameters:
        -----------
        fs : float
            Sampling frequency of the PPG signal in Hz (default: 64Hz)
        """
        self.fs = fs
    def __call__(self, tensor):
        # Apply preprocessing to each channel if it's PPG
        if len(tensor.shape) == 2 and tensor.shape[1] == 1:  # Single-channel PPG
            # Fix: Create a contiguous copy of the numpy array
            numpy_data = tensor.detach().numpy().copy() if tensor.requires_grad else tensor.numpy().copy()
            processed = preprocess_ppg(numpy_data.flatten(), fs=self.fs)
            processed_tensor = torch.FloatTensor(processed)
            return processed_tensor.view(tensor.shape)
        return tensor






def sax_tokenizer(time_series, alphabet_size=4, word_length=4):
    # Normalize the time series
    normalized_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    # Calculate the breakpoints for the alphabet
    breakpoints = np.array([-0.67, 0, 0.67])  # For alphabet_size=4
    if alphabet_size != 4:
        breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    
    # Initialize the symbolic representation
    symbolic_representation = []
    
    # Divide the time series into segments of word_length
    for i in range(0, len(normalized_series), word_length):
        segment = normalized_series[i:i + word_length]
        
        # Calculate the mean of the segment
        segment_mean = np.mean(segment)
        
        # Determine the symbol for this segment
        symbol = np.sum(segment_mean > breakpoints)

        ### Shift this by 1 to reserver 0 for the missing values
        symbol = symbol + 1        
        # Append the symbol to the symbolic representation
        symbolic_representation.append(symbol)
    
    return symbolic_representation


def random_modality_dropout(x, modalities, drop_prob=0.3):
    """
    x: [B, T, D] input tensor
    modalities: list of (start_idx, end_idx) for each modality in feature dimension
    drop_prob: probability to drop (zero out) a modality
    """
    B, T, D = x.shape
    device = x.device
    mask = torch.ones((B, 1, D), device=device)
    for start, end in modalities:
        if torch.rand(1).item() < drop_prob:
            mask[:, :, start:end] = 0
    return x * mask