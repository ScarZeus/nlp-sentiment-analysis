from src.preprocessing import (
    load_data,
    preprocess_data,
    vectorize_data,
    encode_data
)
from src.train import tokenize_text
from sklearn.model_selection import train_test_split

def main():
    file_path = "data/IMDB_Dataset.csv"
    data = load_data(file_path)
    data = preprocess_data(data)

    X = data["review"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    X_train_vec, X_test_vec = vectorize_data(X_train, X_test)

    y_train_enc, y_test_enc = encode_data(y_train, y_test)


    train_tokens, test_tokens = tokenize_text(X_train,X_test)

if __name__ == "__main__":
    main()
