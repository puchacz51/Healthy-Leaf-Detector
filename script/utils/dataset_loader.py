from utils.dictionary import data_path
import os
from utils.split import split
from utils.dataset_downloader import DatasetDownloader


def prepare_data():
    if not os.path.exists(data_path.original) or not os.listdir(data_path.original):
        downloader = DatasetDownloader()
        os.makedirs(data_path.original, exist_ok=True)
        downloader.download_blob(data_path.dataset_path)
    else:
        print("Dane zostały już pobrane.")

#  check if divided data exists without errors on not fount
    if not os.path.exists(data_path.train_healthy) or not os.listdir(data_path.train_healthy): 
        split()
    else:
        print("Dane zostały już podzielone.")

