
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

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


class MIMIC(object):
    def __init__(self):
        ## Minimum cfgs to enable quick baseline runs for TMLR rebuttal
        super(MIMIC, self).__init__()
        self.num_classes = 6
        self.num_modalities = 17
        self.modalities = ['glasgow','BP','HR','Temp','oxy','urine','urea','wbc','bdc2','Na','K','Bil', 'Age','icd9','hem_mal','cancer','adm_type']
        self.sampling_rates = {
            'glasgow': 1,
            'BP': 1,
            'HR': 1,
            'Temp': 1,
            'oxy': 1,
            'urine': 1,
            'urea': 1,
            'wbc': 1,
            'bdc2': 1,
            'Na' :1,
            'K' :1,
            'Bil' :1,
            'Age': 1,
            'icd9': 1,
            'hem_mal': 1,
            'cancer': 1,
            'adm_type': 1,
            # 'Age': 1/24,
            # 'icd9': 1/24,
            # 'hem_mal': 1/24,
            # 'cancer': 1/24,
            # 'adm_type': 1/24,
        }
        
        self.variates = {
            'glasgow': 1,
            'BP': 1,
            'HR': 1,
            'Temp': 1,
            'oxy': 1,
            'urine': 1,
            'urea': 1,
            'wbc': 1,
            'bdc2': 1,
            'Na' :1,
            'K' :1,
            'Bil' :1,
            # Static features
            'Age': 1,
            'icd9': 1,
            'hem_mal': 1,
            'cancer': 1,
            'adm_type': 1,
        }
        self.min_sample_rate = 1
        self.max_sample_rate = 1
        self.base_sample_rate = 1
        self.duration = 24 # 24 hours
        # self.train_set = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
        # self.val_set = ['S16', 'S17']
        # self.eval_set = ['S13', 'S14', 'S15']





class WESAD(object):
    ## classes : 0, 1, 2
    def __init__(self):
        ## Minimum cfgs to enable quick baseline runs for TMLR rebuttal
        super(WESAD, self).__init__()
        self.num_classes = 3
        self.num_modalities = 10
        self.modalities = ['chest_ACC', 'chest_ECG', 'chest_EMG', 'chest_RESP', 'chest_EDA', 'chest_TEMP', 'wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP']
        self.sampling_rates = {
            'chest_ACC': 700,
            'chest_ECG': 700,
            'chest_EMG': 700,
            'chest_RESP': 700,
            'chest_EDA': 700,
            'chest_TEMP': 700,
            'wrist_ACC': 32,
            'wrist_BVP': 64,
            'wrist_EDA': 4,
            'wrist_TEMP': 4,
        }
        
        self.variates = {
            'chest_ACC': 3,
            'chest_ECG': 1,
            'chest_EMG': 1,
            'chest_RESP': 1,
            'chest_EDA': 1,
            'chest_TEMP': 1,
            'wrist_ACC': 3,
            'wrist_BVP': 1,
            'wrist_EDA': 1,
            'wrist_TEMP': 1,
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 700
        self.base_sample_rate = 32
        self.duration = 8 # seconds
        self.train_set = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
        self.val_set = ['S16', 'S17']
        self.eval_set = ['S13', 'S14', 'S15']



class DSADS(object):
    """
    DSADS Dataset Configuration

    Activities:
        1. Sitting
        2. Standing
        3. Lying on back
        4. Lying on right side
        5. Ascending stairs
        6. Descending stairs
        7. Standing in an elevator (still)
        8. Moving around in an elevator
        9. Walking in a parking lot
        10. Walking on a treadmill (flat)
        11. Walking on a treadmill (inclined)
        12. Running on a treadmill (flat)
        13. Exercising on a stepper
        14. Exercising on a cross trainer
        15. Cycling on an exercise bike (horizontal)
        16. Cycling on an exercise bike (vertical)
        17. Rowing
        18. Jumping
        19. Playing basketball
    """

    def __init__(self):
        super(DSADS, self).__init__()
        
        # Dataset Configuration
        self.num_classes = 19
        self.num_modalities = 5
        self.sampling_rate = 25
        self.sequence_len = 125
        self.input_channels = 45
        self.normalize = False
        
        # Sensor Modalities
        self.modalities = [
            'torso',
            'right_arm',
            'left_arm',
            'right_leg',
            'left_leg'
        ]
        self.sampling_rates = {
            'torso' : 25,
            'right_arm' : 25,
            'left_arm' : 25,
            'right_leg' : 25,
            'left_leg' : 25
        }
        self.variates = {
            'torso' : 9,
            'right_arm' : 9,
            'left_arm' : 9,
            'right_leg' : 9,
            'left_leg' : 9
        }
        self.min_sample_rate = 25
        self.max_sample_rate = 25
        self.base_sample_rate = 25
        self.duration = 5 # seconds


class DaliaHAR(object):
    def __init__(self):
        ## Minimum cfgs to enable quick baseline runs for TMLR rebuttal
        super(DaliaHAR, self).__init__()
        self.num_classes = 7
        self.num_modalities = 5
        self.modalities = ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC'] 
        self.sampling_rates = {
        'chest_ACC': 700,
        'wrist_ACC': 32,
        'wrist_BVP': 64,
        'wrist_EDA': 4,
        'wrist_TEMP': 4
        }
        self.variates = {
        'chest_ACC': 3,
        'wrist_ACC': 3,
        'wrist_BVP': 1,
        'wrist_EDA': 1,
        'wrist_TEMP': 1
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 700
        self.base_sample_rate = 32
        self.duration = 8 # seconds
        self.train_set = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
        self.val_set = ['S11', 'S12']
        self.eval_set = ['S13', 'S14', 'S15']


class DaliaHR(object):
    def __init__(self):
        ## Minimum cfgs to enable quick baseline runs for TMLR rebuttal
        super(DaliaHR, self).__init__()
        self.num_classes = 1
        self.num_modalities = 5
        self.modalities = ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC']
        self.sampling_rates = {
        'chest_ACC': 700,
        'wrist_ACC': 32,
        'wrist_BVP': 64,
        'wrist_EDA': 4,
        'wrist_TEMP': 4
        }
        self.variates = {
        'chest_ACC': 3,
        'wrist_ACC': 3,
        'wrist_BVP': 1,
        'wrist_EDA': 1,
        'wrist_TEMP': 1
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 700
        self.base_sample_rate = 64
        self.duration = 8 # seconds
        self.train_set = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
        self.val_set = ['S11', 'S12']
        self.eval_set = ['S13', 'S14', 'S15']
