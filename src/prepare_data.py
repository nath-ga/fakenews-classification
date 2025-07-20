# src/prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# === Pfade & Parameter ===
# DATA_PATH = "data/fakenews.csv"
# nur die Daten für Klimarelevante Themen
DATA_PATH = "data/climate_news.csv"
MAX_LENGTH = 512
TEST_SIZE = 0.2
RANDOM_SEED = 42

# === Daten einlesen ===
df = pd.read_csv(DATA_PATH)

# === Texte & Labels extrahieren ===
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# === Train/Test-Split ===
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
)

# === Tokenizer laden ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# === Tokenisierung ===
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)

# === Ausgabe zur Kontrolle ===
print(f"Train: {len(X_train)} Texte")
print(f"Test:  {len(X_test)} Texte")
print(f"Beispiel Tokens: {train_encodings['input_ids'][0][:10]}")

from fakenews_dataset import FakeNewsDataset

# Dataset-Objekte erzeugen
train_dataset = FakeNewsDataset(train_encodings, y_train)
test_dataset = FakeNewsDataset(test_encodings, y_test)

print(f"\nTrain-Sample (0): {train_dataset[0]}")
print(f"Keys: {train_dataset[0].keys()}")

def get_test_data():
    return test_encodings, y_test
