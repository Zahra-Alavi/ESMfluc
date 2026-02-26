import pandas as pd
import ast

train = pd.read_csv("./data/train_data.csv")
test  = pd.read_csv("./data/test_data.csv")

train["neq"] = train["neq"].apply(ast.literal_eval)
test["neq"]  = test["neq"].apply(ast.literal_eval)

def count_bins(df):
    eq1 = 0
    between_1_2 = 0
    between_2_3 = 0
    above_3 = 0

    for lst in df["neq"]:
        for v in lst:
            v = float(v)
            if abs(v - 1.0) < 1e-9:
                eq1 += 1
            elif 1.0 < v <= 2.0:
                between_1_2 += 1
            elif 2.0 < v <= 3.0:
                between_2_3 += 1
            elif v > 3.0:
                above_3 += 1

    return eq1, between_1_2, between_2_3, above_3


print("Train bins:")
print(count_bins(train))

print("Test bins:")
print(count_bins(test))