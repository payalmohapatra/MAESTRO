'''
Author : Payal Mohapatra
Date Created : 28 March 2025
Project : Group Interaction Modeling in Heterogeneous Irregular Time Series

Description : Use the 8/2 sliding winow/overlap ratio to prepare the dataset for HR prediction task.
'''

import os
import pickle
import torch
import numpy as np
import pandas as pd
import re

# Activity lookup table
ACTIVITY_LOOKUP = {
    # "BASELINE": 0,
    "STAIRS": 0,
    "SOCCER": 1,
    "CYCLING": 2,
    "DRIVING": 3,
    "LUNCH": 4,
    "WALKING": 5,
    "WORKING": 6
}

def segment_data(signal, sampling_rate, window_length=8, window_shift=2):
    """
    Segments the signal into overlapping windows.

    Args:
        signal (array): The input signal.
        sampling_rate (int): Sampling rate of the signal.
        window_length (int): Length of each window in seconds.
        window_shift (int): Shift between consecutive windows in seconds.

    Returns:
        list: List of segmented windows.
    """
    segment_length = int(window_length * sampling_rate)
    segment_shift = int(window_shift * sampling_rate)
    segments = []
    for start in range(0, len(signal) - segment_length + 1, segment_shift):
        segment = signal[start:start + segment_length]
        segments.append(segment)
    return segments

def process_subject(input_file, output_dir, subject_id, activity_name):
    """
    Processes data for a single subject and saves segmented data.

    Args:
        input_file (str): Path to the activity's pickle file.
        output_dir (str): Directory to save processed segments.
        subject_id (int): Subject ID.
        activity_name (str): Name of the activity file being processed.
    """
    # Create subject-specific folder (NO activity-wise subdirectories)
    subject_output_dir = os.path.join(output_dir, f'S{subject_id}')
    os.makedirs(subject_output_dir, exist_ok=True)

    # Load data
    data = pd.read_pickle(input_file)

    # Define sampling rates for each modality
    sampling_rates = {
        'chest_ACC': 700,
        'wrist_ACC': 32,
        'wrist_BVP': 64,
        'wrist_EDA': 4,
        'wrist_TEMP': 4
    }

    # Remove file extension and extract activity name
    filename_without_ext = os.path.splitext(activity_name)[0]
    match = re.search(r'[^_\W]+$', filename_without_ext)  # Matches the last alphanumeric word
    activity_key = match.group(0).upper() if match else ""

    # Get activity label from lookup table
    label = ACTIVITY_LOOKUP.get(activity_key, -1)
    if label == -1:
        print(f"Warning: Activity '{activity_key}' not found in ACTIVITY_LOOKUP! Skipping...")
        return

    # Segment each modality
    segmented_data = {}
    for modality, rate in sampling_rates.items():
        location, sensor = modality.split('_')
        signal = data[location][sensor]
        segmented_data[modality] = segment_data(signal, rate)

    # Save each segment directly in subject folder
    num_segments = len(segmented_data['chest_ACC'])
    
    for i in range(num_segments):
        if all(len(segmented_data[modality]) > i for modality in sampling_rates.keys()):
            segment_dict = {
                'chest_ACC': torch.tensor(segmented_data['chest_ACC'][i]),
                'wrist_ACC': torch.tensor(segmented_data['wrist_ACC'][i]),
                'wrist_BVP': torch.tensor(segmented_data['wrist_BVP'][i]),
                'wrist_EDA': torch.tensor(segmented_data['wrist_EDA'][i]),
                'wrist_TEMP': torch.tensor(segmented_data['wrist_TEMP'][i]),
                'label': torch.tensor(label)  # Store activity class
            }
            torch.save(segment_dict, os.path.join(subject_output_dir, f'{activity_key}_seg{i}.pt'))

    print(f"Created {num_segments} segment files for {activity_name} of subject S{subject_id}.")

def main(base_dataset_folder, output_directory):
    """
    Main function to process all subjects and their activity files.

    Args:
        base_dataset_folder (str): Base folder containing dataset.
        output_directory (str): Directory to save processed data.
    """
    for subject_id in range(1, 16):  # Iterate over all subjects
        subject_folder = os.path.join(base_dataset_folder, f'S{subject_id}')
        
        if not os.path.exists(subject_folder):
            print(f"Folder not found for subject S{subject_id}. Skipping...")
            continue
        
        # Get all .pkl files for the current subject
        activity_files = [f for f in os.listdir(subject_folder) if f.endswith('.pkl')]
        
        if not activity_files:
            print(f"No .pkl files found for subject S{subject_id}. Skipping...")
            continue
        
        for activity_file in activity_files:  # Process each activity file
            activity_file_path = os.path.join(subject_folder, activity_file)
            process_subject(activity_file_path, output_directory, subject_id, activity_file)

if __name__ == "__main__":
    base_dataset_folder = '/home/payal/HeteroIrregTS/data/DaLia/processed_activity_splits_new/'
    output_directory = '/home/payal/HeteroIrregTS/data/DaLia/processed_dalia_activity/'
    
    main(base_dataset_folder, output_directory)
