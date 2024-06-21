import os

class PathConfig:
    def __init__(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        root_path = current_path.split("\\")[:-2] 
        root_path = "\\".join(root_path)
        

        
        self.root = os.getenv('ROOT_PATH', root_path)
        
        self.train = os.getenv('TRAIN_DATA_PATH', 'data/train')
        self.history = os.getenv('HISTORY_PATH', 'history')
        self.home = os.getenv('MODELS_PATH', './model_trained')
        self.test = os.getenv('TEST_DATA_PATH', 'data/test')
        self.model_storage = os.getenv('MODEL_STORAGE_PATH', 'model_trained')
        self.dataset_path = os.getenv('DATASET_PATH', 'data/orginal')

        self.original = os.getenv('ORIGINAL_DATA_PATH', 'data/original')

        self.processed_home = os.getenv('PROCESSED_DATA_PATH', 'data/processed')
        self.processed_healthy = os.getenv('PROCESSED_HEALTHY_DATA_PATH', 'data/processed/healthy')
        self.processed_disease = os.getenv('PROCESSED_DISEASE_DATA_PATH', 'data/processed/disease')

        self.train = os.getenv('TRAIN_DATA_PATH', 'data/train')
        self.train_healthy = os.getenv('TRAIN_HEALTHY_DATA_PATH', 'data/train/healthy')
        self.train_disease = os.getenv('TRAIN_DISEASE_DATA_PATH', 'data/train/disease')

        self.val = os.getenv('VAL_DATA_PATH', 'data/val')
        self.val_healthy = os.getenv('VAL_HEALTHY_DATA_PATH', 'data/val/healthy')
        self.val_disease = os.getenv('VAL_DISEASE_DATA_PATH', 'data/val/disease')

        self.azure_account_url = os.getenv('AZURE_ACCOUNT_URL', 'https://datasetkaggle.blob.core.windows.net/')
        self.azure_account_key = os.getenv('AZURE_SAS_TOKEN', 'sp=r&st=2024-06-15T23:03:18Z&se=2024-06-16T07:03:18Z&spr=https&sv=2022-11-02&sr=b&sig=KmCJJfM7zDXEcWIeWWm0DPGGd%2BV0BxJ%2BCu535IS9wXY%3D')
        self.azure_container_name = os.getenv('AZURE_CONTAINER_NAME', 'dataset')
        self.azure_blob_name = os.getenv('AZURE_BLOB_NAME', 'original.zip')



data_path = PathConfig()
models_path = PathConfig()