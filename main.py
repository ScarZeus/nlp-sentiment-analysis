from src.preprocessing import load_data, preprocess_data
import pandas as pd
def main():
    file_path = "data/IMDB_Dataset.csv"
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    print(processed_data.head())

if __name__=="__main__":
    main()