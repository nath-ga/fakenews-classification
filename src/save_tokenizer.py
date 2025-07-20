# save_tokenizer.py Nachholen, jetzt auch in train_model enthalten. 

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("outputs/model_final")

print("Tokenizer gespeichert.")
