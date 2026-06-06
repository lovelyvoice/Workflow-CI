import pandas as pd
import os
from sklearn.model_selection import train_test_split

print("Downloading Banknote Authentication dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ["variance", "skewness", "curtosis", "entropy", "class"]
df = pd.read_csv(url, names=columns)

print(f"Dataset downloaded. Shape: {df.shape}")

# Buat folder
os.makedirs("banknote_preprocessing", exist_ok=True)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

# Simpan ke csv
train_df.to_csv("banknote_preprocessing/banknote_train.csv", index=False)
test_df.to_csv("banknote_preprocessing/banknote_test.csv", index=False)

print("Berhasil menyimpan banknote_train.csv dan banknote_test.csv di folder banknote_preprocessing")
