import os
import zipfile
from azure.storage.blob import BlobClient

class DatasetDownloader:
    def __init__(self):
        self.azure_blob_url = os.getenv(
            'AZURE_BLOB_URL', 
            'https://datasetkaggle.blob.core.windows.net/dataset/original.zip?sp=r&st=2024-06-16T08:50:41Z&se=2024-06-16T16:50:41Z&spr=https&sv=2022-11-02&sr=b&sig=5u5GAPNcpkKS5wlRup8RdZOZ6t8ItO%2BciyHWJF4urdQ%3D'
        )
        self.azure_model_blob_key = os.getenv(
            'AZURE_MODEL_BLOB_KEY',
            'sp=ra&st=2024-06-16T17:22:47Z&se=2024-06-21T01:22:47Z&spr=https&sv=2022-11-02&sr=c&sig=ZOtSJpgEy3ER2wi971kDZa9gc4mwYcfOTgS1KI7hTuM%3D'
        )
        self.azure_model_blob_url = os.getenv(
            'AZURE_MODEL_BLOB_URL',
            'https://datasetkaggle.blob.core.windows.net/models')
    
    def download_blob(self, download_file_path):
        # Utwórz klienta BlobClient za pomocą pełnego URL z SAS tokenem
        blob_client = BlobClient.from_blob_url(self.azure_blob_url)
        print(f"Pobieranie z {blob_client.url}")

        with open(download_file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
        
        print(f"Pobieranie zakończone! Plik zapisano jako {download_file_path}")
        
        self.unzip_file(download_file_path)
        
        os.remove(download_file_path)
        print(f"Plik ZIP {download_file_path} został usunięty.")
    
    def unzip_file(self, file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
        print(f"Plik {file_path} został rozpakowany.")
        
    def upload_image_to_blob(self, image_path, blob_name):
        print(self.azure_model_blob_url + "/" + blob_name + "?" + self.azure_model_blob_key)
        blob_client = BlobClient.from_blob_url(self.azure_model_blob_url + "/" + blob_name + "?" + self.azure_model_blob_key)
        

        connect_str = "your_connection_string"  # zamień na swoje dane połączenia
        container_name = "your_container_name"  # zamień na nazwę swojego kontenera
        local_file_path = r"C:\Users\jur-s\OneDrive\Pulpit\Martyna\WTUM\model_trained\modelCNN.pth"
        blob_name = "modelCNN.pth"

        # Tworzenie klienta serwisu Blob
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # Tworzenie klienta kontenera
        container_client = blob_service_client.get_container_client(container_name)

        # Tworzenie klienta bloba
        blob_client = container_client.get_blob_client(blob_name)
        # Sprawdzenie, czy plik istnieje i mamy do niego dostęp
        if not os.path.exists(image_path):
            print(f"Błąd: Plik {image_path} nie istnieje.")
            return

        if not os.path.isfile(image_path):
            print(f"Błąd: {image_path} nie jest plikiem.")
            return

        if not os.access(image_path, os.R_OK):
            print(f"Błąd: Brak uprawnień do odczytu pliku {image_path}.")
            return

        try:
            with open(image_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, timeout= 3600)
            print(f"Obraz {image_path} został przesłany do Azure Blob Storage jako {blob_name}.")
        except Exception as e:
            print(f"Błąd podczas przesyłania pliku: {e}")