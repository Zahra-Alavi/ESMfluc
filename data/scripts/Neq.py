#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:55:14 2024

@author: zalavi
"""


import os
import pandas as pd
import glob
import numpy as np

# Directory containing the Neq.tsv files
input_dir = os.path.join(os.getcwd(), 'Neq_data')
output_file = 'average_Neq_results_with_sequences.csv'

# Dictionaries to store the average Neq lists and sequences
average_Neq_dict = {}
sequences_dict = {}

# Process each Neq.tsv file
for tsv_file in glob.glob(os.path.join(input_dir, '*_Neq.tsv')):
    # Extract the protein name from the file name
    protein_name = os.path.basename(tsv_file).replace('_Neq.tsv', '')
    
    # Read the tsv file
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Extract the sequence from the first column
    sequence = df.iloc[:, 0].tolist()
    
    # Compute the average Neq across the three runs
    df['Average_Neq'] = df[['Neq_R1', 'Neq_R2', 'Neq_R3']].mean(axis=1)
    
    # Store the average Neq list and sequence in the dictionaries
    average_Neq_dict[protein_name] = df['Average_Neq'].tolist()
    sequences_dict[protein_name] = sequence

# Convert the average Neq dictionary to a DataFrame
average_Neq_df = pd.DataFrame.from_dict(average_Neq_dict, orient='index').transpose()

# Convert the sequences dictionary to a DataFrame
sequences_df = pd.DataFrame.from_dict(sequences_dict, orient='index').transpose()

# Combine the average Neq and sequences DataFrames
combined_df = pd.concat([average_Neq_df, sequences_df], axis=1, keys=['Neq', 'Sequence'])

# Save the results to a CSV file
combined_df.to_csv(output_file)

print(f'Average Neq results with sequences saved to {output_file}')

# Preparing the data for training
sequences = sequences_df.values
neq_values = average_Neq_df.values

# Creating a DataFrame for sequences and Neq lists
training_data = pd.DataFrame({
    'sequence': [''.join([aa for aa in seq if pd.notna(aa)]) for seq in sequences.T],
    'neq': [list(neq[~np.isnan(neq)]) for neq in neq_values.T]
})

# Save for later use
training_data.to_csv('neq_training_data.csv', index=False)
