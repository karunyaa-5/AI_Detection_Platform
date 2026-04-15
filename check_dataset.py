import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Count labels
print(df['label'].value_counts())