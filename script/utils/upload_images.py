import requests
import os
from utils.dictionary import data_path
def upload_model(base_url, sas_token,  model_name):
    """
    Przesyła model na serwer Azure Blob Storage.

    :param base_url: Podstawowy adres URL do Azure Blob Storage.
    :param sas_token: Token SAS do autoryzacji.
    :param models_path: Ścieżka do katalogu z modelami.
    :param model_name: Nazwa pliku modelu do przesłania.
    """
    model_path = os.path.join(data_path.root, data_path.model_storage, model_name)
    print(data_path.root, data_path.model_storage, model_name)
    # Pełny URL z nazwą pliku
    url = f"{base_url}/{model_name}?{sas_token}"
    
    # Wyświetl nazwę modelu
    print(f"Przesyłanie modelu {model_path} na serwer.")
    # Rozmiar modelu
    print(f"Rozmiar modelu: {os.path.getsize(model_path)} bajtów")
    
    with open(model_path, 'rb') as file:
        # Ustaw nagłówki
        headers = {
            "Content-Type": "application/octet-stream",
            "x-ms-blob-type": "BlockBlob"
        }
        # Prześlij plik do URL-a
        response = requests.put(url, data=file, headers=headers)

    if response.status_code == 201:
        print("Plik został pomyślnie przesłany.")
    else:
        print(f"Wystąpił błąd podczas przesyłania pliku: {response.status_code}")
        print(response.text)

# Przykład użycia funkcji
# base_url = "https://datasetkaggle.blob.core.windows.net/models"
# sas_token = "sp=racwdl&st=2024-06-21T06:42:47Z&se=2024-06-30T14:42:47Z&spr=https&sv=2022-11-02&sr=c&sig=Y2%2BzjQGa6tPMr7TjkjfdyL0Om%2FCYaXa2ZgWvhKGqSd8%3D"
# models_path = "model_trained"
# model_name = "przyklad_modelu.pkl"  # Zmień na rzeczywistą nazwę pliku

# upload_model(base_url, sas_token, models_path, model_name)
