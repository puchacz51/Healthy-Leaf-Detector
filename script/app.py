import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models_schema.modelCNN import LeafClassifier as CNNLeafClassifier
from models_schema.modelResNet import ResNetModel as ResNetLeafClassifier
import os
# Define the model path variable
models_path = {'home': './model_trained'}

def classify_leaf(image, model_name):
    # Definicja transformacji dla obrazów wejściowych
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(models_path["home"])
    print(os.listdir(models_path["home"]))

    # Wybór modelu na podstawie argumentu model_name
    if "resnet" in model_name.lower():
        LeafClassifier = ResNetLeafClassifier
        model_path = f'{models_path["home"]}/{model_name}'
    else:
        LeafClassifier = CNNLeafClassifier
        model_path = f'{models_path["home"]}/{model_name}'

    # check if model exists
    if not os.path.exists(model_path):
        return f"Model {model_name} not found."
    
    # Konwertowanie obrazu do postaci tensora
    image = Image.fromarray(image)
    tensor_image = transform(image).unsqueeze(0)
    
    # Wczytanie modelu
    model = torch.load(model_path)
    model.eval()
    
    # Predykcja
    with torch.no_grad():
        output = model(tensor_image)
    
    if output.item() > 0.5:
        return "Liść wygląda na zdrowy."
    else:
        return "Podejrzewam chorobę liścia."

# Lista dostępnych modeli do wyboru w UI
model_options = [
    "modelCNN.pth",
    "modelResNet.pth"
]

# Tworzenie interfejsu Gradio z wyborem modelu
iface = gr.Interface(
    fn=classify_leaf, 
    inputs=[gr.Image(type="numpy", label="Wrzuć zdjęcie liścia"), gr.Dropdown(choices=model_options, label="Wybierz model")],
    outputs="text", 
    title="Detekcja chorób liści", 
    description="Wrzuć zdjęcie liścia, a model spróbuje określić, czy jest zdrowy czy chory.",
)

# Uruchomienie interfejsu z publicznym linkiem
iface.launch(share=True)
