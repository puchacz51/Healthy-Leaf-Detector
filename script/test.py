import os
from utils.dataset_downloader import DatasetDownloader

DatasetDownloader = DatasetDownloader()

models_path = "model_trained"
# Wszystkie modele w katalogu
models = os.listdir(models_path)
# Ścieżka główna
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Ścieżki bezwzględne do modeli
absolute_models_path = [os.path.join(root_path, models_path, model) for model in models]

for model in absolute_models_path:
    # Nazwa pliku
    model_name = os.path.basename(model)
    # Pełny URL z nazwą pliku
    print(f"Przesyłanie modelu {model} na serwer.")
    # Rozmiar modelu
    print(f"Rozmiar modelu: {os.path.getsize(model)} bajtów")
    DatasetDownloader.upload_image_to_blob( model_name)