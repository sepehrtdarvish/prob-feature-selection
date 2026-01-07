import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_dataset(
    dataset_name = "shashanknecrothapa/ames-housing-dataset",
    dataset_file_path = "AmesHousing.csv"
    ):

    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    dataset_name,
    dataset_file_path,
    )
    
    return df