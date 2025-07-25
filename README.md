# Fake News Klassifikation mit DistilBERT

Dieses Projekt demonstriert, wie man mit Hilfe eines feinjustierten BERT-Modells (DistilBERT) automatisch zwischen echten und gefälschten Nachrichten unterscheidet. Ziel ist es, ein Machine-Learning-Modell zu entwickeln, das gesellschaftlich relevante Fake News erkennt.

## Projektüberblick

- **Modell:** DistilBERT (transformers)
- **Trainingsdaten:** [Kaggle Fake News Dataset (Rahul Thorat)](https://www.kaggle.com/datasets/ioanacheres/disinformation-dataset)
- **Ziel:** Textklassifikation → `0 = real`, `1 = fake`
- **Sprache:** Englisch
- **Anwendungsfall:** Erkennung von Fake News aus News-Artikeln

## Implementierung

- Analyse des Datensatzes (`prepare_data.py`)
- Datenvorverarbeitung (`prepare_data.py`)
- Modelltraining (`train_model.py`)
- Evaluation & Visualisierung (`eval_visualize_metrics.py`)
- Ausgabe von Beispielvorhersagen inkl. Konfidenzwerten (`predict_examples.py`)

## Ergebnisse

- **Test-Genauigkeit:** ca. 91 %
- **Klassifikationsbericht:**  
  Precision, Recall und F1-Score für beide Klassen (siehe Confusion Matrix)
- **Beispielvorhersagen mit Konfidenzwerten:**  
  Zeigt sichere und unsichere Klassifikationen

## 📦 Ordnerstruktur

```
fakenews-classification/
│
├── data/                     # Datensätze
│   └── fakenews.csv
│
├── outputs/                  # Modell, Grafiken, Ergebnisse
│   └── model_final/
│   └── figures/
│
├── src/                      # Quellcode
│   ├── eda.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── eval_visualize_metrics.py
│   └── predict_examples.py
│
└── README.md                 # Dieses Dokument
```

## Anforderungen

pip install transformers datasets torch scikit-learn matplotlib pandas
Getestet unter Python 3.12, ohne GPU

## Modell

distilbert-base-uncased
2 Epochen, Lernrate 5e-5
80/20 Train/Test-Split

## Projekt-Präsentation

Die kompakte Präsentation des Projekts als Bilderserie:

![Slide 1](presentation/1.png)
![Slide 2](presentation/2.png)
![Slide 3](presentation/3.png)
![Slide 4](presentation/4.png)
![Slide 5](presentation/5.png)
![Slide 6](presentation/6.png)

## License

This project is provided under the following terms:

- All **non-code content** (texts, project idea, images, visualizations) is licensed under **CC BY-NC 4.0**  
  → https://creativecommons.org/licenses/by-nc/4.0/

- The **source code** is free to use and adapt for **non-commercial purposes**, with attribution.  
  Commercial use requires permission: [your email]

© 2025 Nathalie Gassert

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

