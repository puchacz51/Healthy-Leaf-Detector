from utils import dataset_loader
from utils.dictionary import data_path

# Download the dataset

with open(data_path.model_storage + "/test.txt", "w") as file:
    file.write("Testowy plik.")
# upload txt file
dataset_loader.upload_model( "test.txt")