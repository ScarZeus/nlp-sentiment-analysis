from transformers import BertTokenizer
import pandas as pd


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_text(train_texts, test_texts):
    train_encodings = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    test_encodings = tokenizer(
        test_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    return train_encodings, test_encodings
