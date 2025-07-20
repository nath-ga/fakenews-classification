# 01_eda.py – EDA für fakenews.csv von Kaggle (Rahul Thorat)

import pandas as pd

DATA_PATH = "data/fakenews.csv"

df = pd.read_csv(DATA_PATH)

print("Spalten:")
print(df.columns)

print(f"\nGesamtanzahl Texte: {len(df)}")
print("\nLabelverteilung (0 = real, 1 = fake):")
print(df['label'].value_counts())

# Beispiele
print("\nBeispiel für echte News:")
print(df[df['label'] == 0]['text'].iloc[0][:500])

print("\nBeispiel für Fake News:")
print(df[df['label'] == 1]['text'].iloc[0][:500])

# Umweltbezug prüfen
keywords = ["climate", "environment", "global warming", "emission", "carbon", "greenhouse", "co2", "iceberg", "storm", "flooding", "surface sealing", "ecologically", "fire"]
climate_df = df[df['text'].str.contains('|'.join(keywords), case=False, na=False)]

print(f"\nAnzahl klimabezogener Nachrichten: {len(climate_df)}")
print(climate_df[['text', 'label']].head(1)['text'].iloc[0][:500])

# Speichern als neuen Datensatz
climate_df.to_csv("data/climate_news.csv", index=False)
print("\n>> Datei gespeichert: data/climate_news.csv")
