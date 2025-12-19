from src.preprocessing import load_data, preprocess_data
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    file_path = "data/IMDB_Dataset.csv"
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    X = processed_data["review"]
    y = processed_data["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=42,shuffle=True,stratify=None)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

if __name__=="__main__":
    main()