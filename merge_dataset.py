import pandas as pd

# -------------------------
# Load Human Dataset
# -------------------------

human_df = pd.read_csv("ASAP2_train_sourcetexts.csv")
print("Human columns:", human_df.columns)

human_df = human_df.rename(columns={"full_text": "text"})
human_df["label"] = 0

# -------------------------
# Load AI Dataset
# -------------------------

ai_df = pd.read_csv("AI Generated Essays Dataset.csv")
print("AI columns:", ai_df.columns)

ai_df["label"] = 1

# -------------------------
# Select Needed Columns
# -------------------------

human_df = human_df[["text", "label"]]
ai_df = ai_df[["text", "label"]]

# -------------------------
# Combine
# -------------------------

dataset = pd.concat([human_df, ai_df], ignore_index=True)

dataset.to_csv("dataset.csv", index=False)

print("Dataset merged successfully!")
print("Total samples:", len(dataset))