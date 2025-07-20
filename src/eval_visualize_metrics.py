# 06_eval_visualize_metrics.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Verzeichnis für die Ausgabe erstellen, falls es nicht existiert
os.makedirs("outputs/figures", exist_ok=True)

# === 1. Konfiguration ===
# MODEL_PATH = "outputs/model_final/"
# DATA_PATH = "data/fakenews.csv"
# Mit Klima Subset
MODEL_PATH = "outputs/model_climate"
DATA_PATH = "data/climate_news.csv"


# === 2. Modell & Tokenizer laden ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# === 3. Daten einlesen ===
df = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# === 4. Tokenisierung ===
inputs = tokenizer(list(X_test), padding=True, truncation=True, return_tensors="pt")

# === 5. Vorhersage (Inference) ===
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).numpy()

# === 6. Auswertung ===
print("\nKlassifikationsbericht:")
print(classification_report(y_test, predictions, target_names=["real", "fake"]))

cm = confusion_matrix(y_test, predictions)

# === 7. Visualisierung ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix Fake News Klassifikation")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix.png")  # optional
plt.show()

# === Unsicherheitsprüfungen
from torch.nn.functional import softmax

print("\n>>> Beispiele mit Konfidenzwerten:")

# Wahrscheinlichkeiten berechnen
probs = softmax(outputs.logits, dim=1)
confs, preds = torch.max(probs, dim=1)

# Liste der Labelnamen
label_names = ["real", "fake"]

# Zeige z. B. die ersten 10 Fälle mit Unsicherheitsprüfung
for i in range(10):
    text_snippet = X_test.iloc[i][:100].replace("\n", " ") + "..."  # Textanfang
    predicted_label = label_names[preds[i]]
    confidence = confs[i].item()

    if confidence < 0.80:
        note = "❓ UNSICHER"
    else:
        note = "✅ SICHER"

    print(f"{note}, {predicted_label} ({confidence:.2f}): {text_snippet}")
