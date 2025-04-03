import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import argparse

def analysis(args):
    df = pd.read_csv(args.folder + "/results.csv")

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    wrong_aa_count_0 = Counter()
    wrong_aa_count_1 = Counter()
    
    wrong_aa_neq_values_1 = defaultdict(list)
    
    total_aa_count_0 = Counter()
    total_aa_count_1 = Counter()

    # Iterate through the dataset

    # columns are: sequence,neq values,pred,true label
    for seq, pred_classes, true_classes, neq_values in zip(df["sequence"], df["pred"], df["true label"], df["neq values"]):
        true_classes = eval(true_classes)
        pred_classes = eval(pred_classes)
        neq_values = eval(neq_values.replace("tensor(", "").replace(")", "")) 
        seq = list(seq)
        for aa, true_c, pred_c, neq_value in zip(seq, true_classes, pred_classes, neq_values):
            if true_c != pred_c:  # If prediction is incorrect
                if true_c == 0:
                    wrong_aa_count_0[aa] += 1
                elif true_c == 1:
                    wrong_aa_count_1[aa] += 1
                    wrong_aa_neq_values_1[aa].append(neq_value)
                    
            # Count total occurrences of each amino acid in the true classes
            if true_c == 0:
                total_aa_count_0[aa] += 1
            elif true_c == 1:
                total_aa_count_1[aa] += 1


    # Plot the histogram
    x = np.arange(len(amino_acids))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, [wrong_aa_count_0[aa] for aa in amino_acids], width, label="True Class is 0 But Predicted Class is 1")
    plt.bar(x + width/2, [wrong_aa_count_1[aa] for aa in amino_acids], width, label="True Class is 1 But Predicted Class is 0")
    plt.bar(x - width/2, [total_aa_count_0[aa] for aa in amino_acids], width, alpha=0.5, label="Total Class 0")
    plt.bar(x + width/2, [total_aa_count_1[aa] for aa in amino_acids], width, alpha=0.5, label="Total Class 1")

    if (not os.path.exists(args.folder + "/plot")):
        os.makedirs(args.folder + "/plot")
    # Labels and title
    plt.xticks(x, amino_acids)
    plt.xlabel("Amino Acids")
    plt.ylabel("Counts of Wrong Predictions")
    plt.title("Amino Acid Distribution in Wrongly Predicted Classes")
    plt.legend()
    plt.show()
    plt.savefig(args.folder + "/plot/wrong_predictions.png")
    plt.close()
    
    # Plot the histogram for neq values for each amino acid
    for aa, neq_values in wrong_aa_neq_values_1.items():
        plt.figure(figsize=(12, 6))  
        bins = np.arange(1, 8, 1)
        plt.hist(neq_values, bins=bins, edgecolor="black", rwidth=0.8)
        plt.title(f"True Neq Values of {aa} That Were Wrongly Classified As 0")
        plt.xlabel("neq values")
        plt.ylabel("Frequency")
        plt.savefig(args.folder + f"/plot/neq_values_{aa}.png")
        plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()
    analysis(args)



