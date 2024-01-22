import os
import opendatasets as od
import json

dataset_URL = "https://www.kaggle.com/datasets/milesh1/35-million-chess-games"
data_folder = "landuse-scene-classification"


def make_dataset():
    """
    Description:
        Downloads the dataset into data and extracts the content using the opendatasets library.
        Important to have the kaggle.json file in the root folder.
    Returns:
        None

    """
    od.download(dataset_URL, data_dir="data/", unzip=True)
    try:
        os.rename('data/35-million-chess-games/all_with_filtered_anotations_since1998.txt', 'data/35-million-chess-games/all_with_filtered_annotations_since1998.txt')
        print('Fixed spelling error')
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    make_dataset()
