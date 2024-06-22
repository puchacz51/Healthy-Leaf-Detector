import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models_schema.modelCNN import LeafClassifier as CNNLeafClassifier
from models_schema.modelResNet import ResNetModel as ResNetLeafClassifier
import os
from azure.storage.blob import BlobServiceClient
# Define the model path variable


azure_storage_account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'datasetkaggle')
azure_storage_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY', 'kk')
azure_model_container_name = os.getenv('AZURE_MODEL_CONTAINER_NAME', 'models')
azure_history_container_name = os.getenv('AZURE_HISTORY_CONTAINER_NAME', 'history')
loaded_model_path = os.getenv('MODEL_PATH', './model_trained')

loaded_models_list =os.listdir(loaded_model_path)


# get all models from azure
blob_service_client = BlobServiceClient(
    account_url=f"https://{azure_storage_account_name}.blob.core.windows.net",
    credential=azure_storage_account_key
)
# get list of available models
def load_models():
    list_of_models = blob_service_client.get_container_client(azure_model_container_name).list_blobs()
    # download all models ended with .pth
    if(not os.path.exists(loaded_model_path)):
        os.makedirs(loaded_model_path, exist_ok=True)

    for model in list_of_models:
        if model.name.endswith('.pth') and loaded_models_list.count(model.name) == 0:
            print(f"Downloading model {model.name}")
            blob_client = blob_service_client.get_blob_client(container=azure_model_container_name, blob=model.name)
            with open(f'{loaded_model_path}/{model.name}', "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            loaded_models_list.append(model.name)





load_models()

def classify_leaf(image, model_name):
    # Definicja transformacji dla obrazów wejściowych
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    # Wybór modelu na podstawie argumentu model_name
    if "resnet" in model_name.lower():
        LeafClassifier = ResNetLeafClassifier
        model_path = f'{loaded_model_path}/{model_name}'
    else:
        LeafClassifier = CNNLeafClassifier
        model_path = f'{loaded_model_path}/{model_name}'

    # check if model exists
    if not os.path.exists(model_path):
        return f"Model {model_name} not found."
    
    # Konwertowanie obrazu do postaci tensora
    image = Image.fromarray(image)
    tensor_image = transform(image).unsqueeze(0)
    
    # Wczytanie modelu
    model = LeafClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predykcja
    with torch.no_grad():
        output = model(tensor_image)
    
    if output.item() > 0.5:
        return "Liść wygląda na zdrowy."
    else:
        return "Podejrzewam chorobę liścia."

# Lista dostępnych modeli do wyboru w UI
# model_options = [
#     "modelCNN.pth",
#     "modelResNet.pth"
# ]
model_options = os.listdir(loaded_model_path)


# Tworzenie interfejsu Gradio z wyborem modelu
iface = gr.Interface(
    fn=classify_leaf, 
    inputs=[gr.Image(type="numpy", label="Wrzuć zdjęcie liścia"), gr.Dropdown(choices=model_options, label="Wybierz model")],
    outputs="text", 
    title="Detekcja chorób liści", 
    description="Wrzuć zdjęcie liścia, a model spróbuje określić, czy jest zdrowy czy chory.",
)

# Uruchomienie interfejsu z publicznym linkiem
iface.launch(share=True,server_port=80)
