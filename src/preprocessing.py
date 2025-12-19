import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path:str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def basic_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    data["review"] = data["review"].str.lower()
    data["review"] = data["review"].str.replace(r"<.*?>", "", regex=True)
    data["review"] = data["review"].str.replace(r"http\S+", "", regex=True)
    data["review"] = data["review"].str.replace(r"\s+", " ", regex=True).str.strip()
    return data


def vectorize_data(data: pdf.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfCevtorizer(max_features = 20000
                                 min_df = 5,
                                 max_df = 0.8)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = basic_cleaning(data)
    return data