import torch
from transformers import EsmModel, EsmTokenizer
import pandas as pd
import ast
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt

model_name = "facebook/esm2_t12_35M_UR50D"
esm_model = EsmModel.from_pretrained(model_name)
esm_tokenizer = EsmTokenizer.from_pretrained(model_name)

def esm_embedding(sequence):
    tokens = esm_tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False, truncation=False)
    with torch.no_grad():
        outputs = esm_model(**tokens)
    return outputs.last_hidden_state

data = pd.read_csv("../../data/train_data.csv")
seq_embed = esm_embedding(data["sequence"][0])

X_train = data["sequence"].apply(esm_embedding)
y_train = data["neq"].apply(ast.literal_eval)

X_train_flat = []
for row in X_train:
  row = row.reshape(-1, 480)
  X_train_flat.extend(row)
y_train_flat = [neq for row in y_train for neq in row]

test_data = pd.read_csv("../../data/test_data.csv")
X_test = test_data["sequence"].apply(esm_embedding)
y_test = test_data["neq"].apply(ast.literal_eval)
X_test_flat = []
for row in X_test:
  row = row.reshape(-1, 480)
  X_test_flat.extend(row)
y_test_flat = [neq for row in y_test for neq in row]

bayesian_model = BayesianRidge()
bayesian_model.fit(X_train_flat, y_train_flat)

import os
if not os.path.exists("../../plot/bayesian/residuals"):
    os.makedirs("../../plot/bayesian/residuals")
if not os.path.exists("../../plot/bayesian/true_vs_predicted"):
    os.makedirs("../../plot/bayesian/true_vs_predicted")
for i in range(len(X_test)):
    x_test_one = X_test[i].reshape(-1, 480)
    y_pred = bayesian_model.predict(x_test_one)
    true_value = y_test[i]

    # Plotting residuals
    residuals = y_pred - true_value
    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted values")
    plt.show()
    plt.savefig(f"../../plot/bayesian/residuals/residuals_{i}.png")
    plt.close()
    
    # Plot the true value vs predicted value
    plt.scatter(y_pred, true_value)
    plt.xlabel("Predicted values")
    plt.ylabel("True values")
    plt.title("True values vs Predicted values")
    plt.show()
    plt.savefig(f"../../plot/bayesian/true_vs_predicted/true_vs_predicted_{i}.png")
    plt.close()

print("Done!")



