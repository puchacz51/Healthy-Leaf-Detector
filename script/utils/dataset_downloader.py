import os
import zipfile
from azure.storage.blob import BlobServiceClient, BlobClient
from utils.dictionary import data_path

class DatasetDownloader:
    def __init__(self):
        self.azure_storage_account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'datasetkaggle')
        self.azure_storage_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY', '')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME', 'dataset')
        self.blob_name = os.getenv('AZURE_BLOB_NAME', 'original.zip')
        self.model_container_name = os.getenv('AZURE_MODEL_CONTAINER_NAME', 'models')
        self.history_container_name = os.getenv('AZURE_HISTORY_CONTAINER_NAME', 'history')

        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.azure_storage_account_name}.blob.core.windows.net",
            credential=self.azure_storage_account_key
        )
    
    def download_blob(self, download_file_path):
        # Create BlobClient using the account name and key
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=self.blob_name)
        print(f"Downloading from {blob_client.url}")

        with open(download_file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
        
        print(f"Download complete! File saved as {download_file_path}")
        
        self.unzip_file(download_file_path)
        
        
        os.remove(download_file_path)
        print(f"ZIP file {download_file_path} has been deleted.")
    
    def unzip_file(self, file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
        print(f"File {file_path} has been unzipped.")
        
    def upload_model(self, file_path, blob_name):
        # Create BlobClient for the destination blob
        blob_client = self.blob_service_client.get_blob_client(container=self.model_container_name, blob=blob_name)
        print(f"Uploading to {blob_client.url}")

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"Upload complete! File {file_path} has been uploaded as {blob_name}.")

    def upload_history(self, file_path, blob_name):
        # Create BlobClient for the destination blob
        blob_client = self.blob_service_client.get_blob_client(container=self.history_container_name, blob=blob_name)
        print(f"Uploading to {blob_client.url}")

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"Upload complete! File {file_path} has been uploaded as {blob_name}.")

# Example usage
downloader = DatasetDownloader()
