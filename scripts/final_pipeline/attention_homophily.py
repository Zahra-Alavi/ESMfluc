#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:11:20 2025

Description: This script investigates attention homophily in the test set. 
@author: zalavi
"""

import pandas as pd
import numpy as np

df = pd.read_json('../../results/output.json')

# one-hot dictionary for ss type
onehot = {
    'C': [1,0,0],
    'H': [0,1,0],
    'E': [0,0,1]
}

def build_attention_info(row):
    seq = row['sequence']
    L = len(seq)

    attn_matrix = row['attention_weights']   # shape (L, L)
    ss_list = row['ss_pred']                # length L

    attention_vectors = [] 
    dot_products = []

    for i in range(L):
        query_ss = onehot[ss_list[i]]  # e.g. [1,0,0] for 'C'
        attn_row = attn_matrix[i]      # length L

        sum_ = [0.0, 0.0, 0.0]
        for j in range(L):
            w_ij = attn_row[j]
            key_ss = onehot[ss_list[j]]
            sum_[0] += w_ij * key_ss[0]
            sum_[1] += w_ij * key_ss[1]
            sum_[2] += w_ij * key_ss[2]

        attention_vectors.append(sum_)

        dot_i = (query_ss[0]*sum_[0] +
                 query_ss[1]*sum_[1] +
                 query_ss[2]*sum_[2])
        dot_products.append(dot_i)

    return pd.Series([attention_vectors, dot_products], 
                     index=['attention_vectors','dot_products'])


df[['attention_vectors','dot_products']] = df.apply(build_attention_info, axis=1)


dot_prods = df['dot_products'] #df['dot_products'][k]: list of length L, each scalar in [0,1]

attention_vectors = df['attention_vectors'] #df['attention_vectors'][k]: list of length L, each element is [C,H,E]


rows = []

for seq_idx in range(len(df)):
    ss_list   = df["ss_pred"][seq_idx]       # e.g. ['C','C','H','E', ...], length L
    dot_list  = df["dot_products"][seq_idx]  # e.g. [0.3,0.7,0.45,0.22, ...], length L
    attn_list = df["attention_vectors"][seq_idx]  # e.g. [[0.5,0.3,0.2], [...], ...], length L
    L = len(ss_list)

    for i in range(L):
        query_type = ss_list[i]    # 'C','H','E'
        dot_i      = dot_list[i]   # fraction of attention on same SS
        attn_vec   = attn_list[i]  # 3D distribution, e.g. [A^C, A^H, A^E]

        # each row in analysis_df is for a single residue
        rows.append((query_type, dot_i, attn_vec))

analysis_df = pd.DataFrame(rows, columns=["query_ss","dot_value","attn_vector"])


analysis_df["A_C"] = analysis_df["attn_vector"].apply(lambda v: v[0])
analysis_df["A_H"] = analysis_df["attn_vector"].apply(lambda v: v[1])
analysis_df["A_E"] = analysis_df["attn_vector"].apply(lambda v: v[2])


mean_vectors = analysis_df.groupby("query_ss")[["A_C","A_H","A_E"]].mean()
print(mean_vectors)


import matplotlib.pyplot as plt

for ss_type, group in analysis_df.groupby("query_ss"):
    plt.figure()
    plt.hist(group["dot_value"], bins=30, alpha=0.7)
    plt.title(f"Distribution of dot_value for query {ss_type}")
    plt.xlabel("dot_value (fraction of attention to same SS)")
    plt.ylabel("Count")
    plt.show()

# one-hot dictionary for Neq class
onehot_neq = {0: [1,0],
              1: [0,1]}

def build_attention_info_neq(row):
    A        = row["attention_weights"]
    neq_list = row["neq_preds"]
    L        = len(neq_list)

    attn_vecs_neq = []
    dots_neq      = []

    for i in range(L):
        q_vec = onehot_neq[neq_list[i]]
        a_vec = [0.0, 0.0]

        for j in range(L):
            w = A[i][j]
            k_vec = onehot_neq[neq_list[j]]
            a_vec[0] += w * k_vec[0]
            a_vec[1] += w * k_vec[1]

        attn_vecs_neq.append(a_vec)
        dots_neq.append(q_vec[0]*a_vec[0] + q_vec[1]*a_vec[1])

    return pd.Series([attn_vecs_neq, dots_neq],
                     index=["attention_vectors_neq",
                            "dot_products_neq"])

df[["attention_vectors_neq","dot_products_neq"]] = (
    df.apply(build_attention_info_neq, axis=1)
)

total_vec_0 = np.zeros(2, dtype=float)  
total_vec_1 = np.zeros(2, dtype=float)   
count_0 = 0       # number of residues with query‑neq = 0
count_1 = 0       # number of residues with query‑neq = 1

for _, row in df.iterrows():
    attn_vecs = row["attention_vectors_neq"]  
    neq_list  = row["neq_preds"]               # list of 0/1

    for vec, q_neq in zip(attn_vecs, neq_list):
        if q_neq == 0:
            total_vec_0 += vec
            count_0     += 1
        else:
            total_vec_1 += vec
            count_1     += 1


mean_vec_0 = total_vec_0 / count_0    
mean_vec_1 = total_vec_1 / count_1     


print("Average attention distribution (query residue has NEQ = 0):")
print(f"  NEQ 0 : {mean_vec_0[0]:.3f}")
print(f"  NEQ 1 : {mean_vec_0[1]:.3f}\n")

print("Average attention distribution (query residue has NEQ = 1):")
print(f"  NEQ 0 : {mean_vec_1[0]:.3f}")
print(f"  NEQ 1 : {mean_vec_1[1]:.3f}")


def received_attention(attn_matrix):
    A = np.asarray(attn_matrix, dtype=float)   # (L,L)
    recv = A.sum(axis=0)          # sum over queries -> shape (L,)
    recv = recv / len(recv)       # divide by L to normalize
    return recv.tolist()

df["received_attention"] = df["attention_weights"].apply(received_attention)

aa20 = list("ACDEFGHIKLMNPQRSTVWY")

# running totals
totals  = {aa: 0.0 for aa in aa20}
counts  = {aa: 0   for aa in aa20}

for _, row in df.iterrows():
    seq   = row["sequence"]
    recv  = row["received_attention"]      # list of length L
    for aa, r in zip(seq, recv):
        if aa in totals:                   
            totals[aa] += r
            counts[aa] += 1

# mean received‑attention per amino‑acid
mean_recv = {aa: totals[aa] / counts[aa] for aa in aa20}

# tidy as a Series for pretty display
mean_recv_series = pd.Series(mean_recv).sort_index()

print("Average fraction of total attention received by each amino acid:\n")
print(mean_recv_series)
