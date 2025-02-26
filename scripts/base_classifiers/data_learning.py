"""
Description: This file contains the DataLearning class which is used to analyze the data.
Date: 2025-02-07
Author: Ngoc Kim Ngan Tran
"""

import os
from utils import *
from decimal import Decimal

class DataLearning:
    def __init__(self, sequences, neq_values):
        self.sequences = sequences
        self.neq_values = neq_values
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        if not os.path.exists("plot/data/neq"):
            os.makedirs("plot/data/neq")

    def analyze_data(self):
        lengths = [len(seq) for seq in self.sequences]
        plot_histogram(lengths, bins=50, title="Histogram of Sequence Lengths", xlabel="Sequence Length", ylabel="Frequency", save_as="plot/data/sequence_lengths.png")
        
        aa_counts = {aa: 0 for aa in self.amino_acids}
        for seq in self.sequences:
            for aa in seq:
                aa_counts[aa] += 1
        print(aa_counts)
        plot_horizontal_bar(list(aa_counts.values()), list(aa_counts.keys()), title="Amino Acid Frequencies", xlabel="Frequency", ylabel="Amino Acids", save_as="plot/data/amino_acid_frequencies.png")
        
        aa_neq = {aa: [] for aa in self.amino_acids}
        for seq, neq_seq in zip(self.sequences, self.neq_values):
            for i, aa in enumerate(seq):
                aa_neq[aa].append(neq_seq[i])
        
        for aa in self.amino_acids:
            plot_histogram(aa_neq[aa], bins=50, title=f"Neq Distribution for {aa}", xlabel="Neq Value", ylabel="Frequency", save_as=f"plot/data/neq/neq_distribution_{aa}.png")
        
        aa_avg_neq = {aa: sum(neq_values) / len(neq_values) for aa, neq_values in aa_neq.items()}
        plot_horizontal_bar(list(aa_avg_neq.values()), list(aa_avg_neq.keys()), title="Average Neq Values for Amino Acids", xlabel="Average Neq Value", ylabel="Amino Acids", save_as="plot/data/neq/average_neq_values.png")