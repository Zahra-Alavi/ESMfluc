import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../data/mdcath/mdcath_neq_dataset.csv")

# Extract the PDB ID (First 4 characters) to prevent leakage
df['pdb_group'] = df['domain'].str[:4]

unique_groups = df['pdb_group'].unique()
train_groups, test_groups = train_test_split(unique_groups, test_size=0.20, random_state=42)

train_df = df[df['pdb_group'].isin(train_groups)].copy()
test_df = df[df['pdb_group'].isin(test_groups)].copy()

train_df.to_csv("../../data/mdcath/train_split.csv", index=False)
test_df.to_csv("../../data/mdcath/test_split.csv", index=False)

print(f"Total domains: {len(df)}")
print(f"Training set: {len(train_df)} domains")
print(f"Test set: {len(test_df)} domains")