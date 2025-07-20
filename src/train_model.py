# src/train_model.py

import transformers
# print("Transformers-Version:", transformers.__version__)

import torch
#from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments
from fakenews_dataset import FakeNewsDataset
from prepare_data import train_encodings, test_encodings, y_train, y_test

# === Schritt 1: Datensätze vorbereiten ===
train_dataset = FakeNewsDataset(train_encodings, y_train)
test_dataset = FakeNewsDataset(test_encodings, y_test)

# === Schritt 2: Modell laden ===
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# === Schritt 3: Trainingsargumente ===
training_args = TrainingArguments(
    output_dir="outputs/checkpoints",      # wohin Checkpoints gespeichert werden
    num_train_epochs=2,                     # wie oft die Daten durchlaufen werden
    per_device_train_batch_size=8,          # Batch-Größe pro Gerät (klein bei CPU!)
    per_device_eval_batch_size=8,
    warmup_steps=50,                        # für Lernratenplanung
    weight_decay=0.01,
    logging_dir="outputs/logs",            # wohin Logs gespeichert werden
    logging_steps=50,                       # alle 50 Schritte eine Ausgabe
    eval_strategy="epoch",                 # eval nach jeder Epoche
    save_strategy="epoch",                # speichern nach jeder Epoche
    load_best_model_at_end=True,           # bestes Modell merken
    metric_for_best_model="accuracy",
    save_total_limit=2                     # max. 2 gespeicherte Modelle
)

# === Schritt 4: Metriken definieren ===
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# === Schritt 5: Trainer-Objekt ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# === Schritt 6: Training starten ===
print("\n>>> Starte Training...")
trainer.train()

# === Schritt 7: Modell speichern ===

# print("\n>>> Speichere trainiertes Modell in outputs/model_final")
# Klima Subset
print("\n>>> Speichere trainiertes Modell in outputs/model_climate")
# trainer.save_model("outputs/model_final")
# Klima Subset
trainer.save_model("outputs/model_climate")

from transformers import AutoTokenizer

# === Schritt 8: Tokenizer speichern ===
# print("\n>>> Speichere Tokenizer in outputs/model_final")
# Klima Subset
print("\n>>> Speichere Tokenizer in outputs/model_climate")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer.save_pretrained("outputs/model_final")
# Klima Subset
tokenizer.save_pretrained("outputs/model_climate")

print("\n>>> Training abgeschlossen.")
