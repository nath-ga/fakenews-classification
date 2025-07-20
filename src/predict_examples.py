# predict_examples.py

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

# === Modell & Tokenizer laden ===
MODEL_DIR = "outputs/model_final/"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# === Beispieltexte (du kannst eigene ergänzen) ===
examples = [
    "NASA confirms 2023 was the hottest year on record.",
    "Celebrity couple breaks up after ten years.",
    "Scientists found evidence that climate change is a hoax.",
    "The government launches a new initiative to plant trees in urban areas."
]

# === Vorhersagefunktion ===
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
    return pred_label, confidence

# === Ergebnisse ausgeben ===
print("📄 Vorhersagen für Beispieltexte:\n")
for text in examples:
    label, conf = predict(text)
    label_name = "REAL" if label == 0 else "FAKE"
    sicherheit = "✅ SICHER" if conf > 0.85 else "❓ UNSICHER"
    print(f"{sicherheit}, {label_name} ({conf:.2f}): {text[:100]}...")

print("\n🔍 Manuelle Eingabe zur Vorhersage (Abbrechen mit leerer Eingabe):")
while True:
    user_input = input("\nGib einen Text ein: ").strip()
    if user_input == "":
        print("Beendet.")
        break

    # Tokenisierung
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

    label_str = "REAL" if pred_class == 0 else "FAKE"
    sicherheit = "✅ SICHER" if confidence >= 0.8 else "❓ UNSICHER"
    print(f"\n{sicherheit}, {label_str} ({confidence:.2f}): {user_input[:70]}{'...' if len(user_input) > 70 else ''}")
