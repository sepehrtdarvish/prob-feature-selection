import pandas as pd


def load_dataset(dataset_file_path = "AmesHousing.csv"):
    return pd.read_csv('./data/AmesHousing.csv')
