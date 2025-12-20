import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def basic_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    data["review"] = data["review"].str.lower()
    data["review"] = data["review"].str.replace(r"<.*?>", "", regex=True)
    data["review"] = data["review"].str.replace(r"http\S+", "", regex=True)
    data["review"] = data["review"].str.replace(r"\s+", " ", regex=True).str.strip()
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    return basic_cleaning(data)

def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=5,
        max_df=0.8
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec


def encode_data(y_train, y_test):
    encoder = LabelEncoder()

    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    return y_train_enc, y_test_enc
