'''
Author : Payal Mohapatra
Date Created : 28 March 2025
Project : Group Interaction Modeling in Heterogeneous Irregular Time Series

Description : Remove nontransient parts and split the data as per the activity 
'''
import os
import pickle
import numpy as np
import pandas as pd

# Define sampling rates for different modalities
sampling_rates = {
    'chest_ACC': 700,
    'wrist_ACC': 32,
    'wrist_BVP': 64,
    'wrist_EDA': 4,
    'wrist_TEMP': 4
}

def extract_activity_splits_with_time(input_file, csv_file, output_dir, subject_id):
    """
    Extracts activity-specific splits for a subject using absolute time from a CSV file.
    Excludes "NoActivity" regions.
    """
    # Load the data from the .pkl file
    data = pd.read_pickle(input_file)

    # Load the absolute time information from the CSV file
    activity_times = pd.read_csv(csv_file)

    signals = data['signal']
    subject_output_dir = os.path.join(output_dir, f'S{subject_id}')
    os.makedirs(subject_output_dir, exist_ok=True)

    # Iterate over activities in the CSV file
    for i in range(len(activity_times) - 1):
        activity_name = activity_times.iloc[i][0]
        start_time = activity_times.iloc[i][1]
        end_time = activity_times.iloc[i + 1][1]
        # print(activity_name)
        # print(start_time)
        # print(end_time)
        # breakpoint()

        # Skip transient periods labeled as "NoActivity"
        if activity_name.lower() == "noactivity":
            continue

        print(f"Processing activity {activity_name} for subject S{subject_id}...")

        # Extract data for this time range
        activity_data = {
            'chest': {},
            'wrist': {},
            'activity': activity_name
        }

        for modality in ['ACC']:
            if modality in signals['chest']:
                modality_key = f'chest_{modality}'
                sampling_rate = sampling_rates[modality_key]
                start_idx = int(start_time * sampling_rate)
                end_idx = int(end_time * sampling_rate)
                activity_data['chest'][modality] = signals['chest'][modality][start_idx:end_idx]

        for modality in ['ACC', 'BVP', 'EDA', 'TEMP']:
            if modality in signals['wrist']:
                modality_key = f'wrist_{modality}'
                sampling_rate = sampling_rates[modality_key]
                start_idx = int(start_time * sampling_rate)
                end_idx = int(end_time * sampling_rate)
                activity_data['wrist'][modality] = signals['wrist'][modality][start_idx:end_idx]

        # Save extracted data as a .pkl file
        output_file = os.path.join(subject_output_dir, f'S{subject_id}_{activity_name}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(activity_data, f)

        print(f"Saved {output_file}.")

def main(base_dataset_folder, output_directory):
    """
    Main function to process all subjects using absolute time from CSV files.
    """
    for subject_id in range(10, 16):  # Subjects S1 to S15
        subject_folder = os.path.join(base_dataset_folder, f'S{subject_id}')
        pkl_file = os.path.join(subject_folder, f'S{subject_id}.pkl')
        csv_file = os.path.join(subject_folder, f'S{subject_id}_activity.csv')

        if os.path.exists(pkl_file) and os.path.exists(csv_file):
            print(f"Processing subject S{subject_id}...")
            extract_activity_splits_with_time(pkl_file, csv_file, output_directory, subject_id)
        else:
            print(f"PKL or CSV file not found for subject S{subject_id}")

if __name__ == "__main__":
    base_dataset_folder = '/home/payal/HeteroIrregTS/data/DaLia/PPG_FieldStudy'
    output_directory = '/home/payal/HeteroIrregTS/data/DaLia/processed_activity_splits_new'

    main(base_dataset_folder, output_directory)
