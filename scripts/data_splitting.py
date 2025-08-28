# This file mainly uses for splitting data for training and testing from neq_original_data.csv to train_data.csv and test_data.csv

from sklearn.model_selection import train_test_split
import pandas as pd

def get_validate_data(data):
    # All the sequences need to have the length < 1024
    return data[data['sequence'].str.len() < 1024]

def split_data(data):
    X = data['sequence']
    y = data['neq']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = pd.DataFrame({'sequence': X_train, 'neq': y_train})
    test_data = pd.DataFrame({'sequence': X_test, 'neq': y_test})
    return train_data, test_data


data = pd.read_csv("../data/neq_original_data.csv")
print(data)
data = get_validate_data(data)
train_data, test_data = split_data(data)

# save to file
train_data.to_csv("../data/train_data.csv", index=False)
test_data.to_csv("../data/test_data.csv", index=False)