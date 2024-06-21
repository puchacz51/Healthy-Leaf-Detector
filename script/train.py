from utils import dataset_loader
from utils import train_model
from models_schema.modelCNN import LeafClassifier
from models_schema.modelResNet import ResNetModel
from utils.prepare_data import get_train_loader
from utils.dictionary import data_path
from utils.train_model import train_model,prepare_model
from utils.history import save_history
from utils .dataset_loader import upload_model
import torch
import torch.optim as optim
import torch.nn as nn
import os 
import time
if __name__ == "__main__":
    dataset_loader.prepare_data()
    print("Dane zostały przygotowane.")
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Urządzenie: {device}")
    

    train_loader = get_train_loader(data_path.train)
    modelResNet = prepare_model(device, ResNetModel)
    modelCNN = prepare_model(device, LeafClassifier)

    optimizerCNN = optim.Adam(modelCNN.parameters(), lr=0.001)
    optimizerResNet = optim.Adam(modelResNet.parameters(), lr=0.001)
    criterionCNN = nn.BCELoss()
    criterionResNet = nn.BCEWithLogitsLoss()


    train_dateils_Resnt = train_model(modelResNet, device,train_loader , criterionResNet, optimizerResNet, 1)
    print("Model ResNet został wytrenowany.")

    train_dateils_CNN =  train_model(modelCNN, device, train_loader, criterionCNN, optimizerCNN, 1)

    print("Model CNN został wytrenowany.")

    # save_history(train_dateils_Resnt, "ResNet")
    save_history(train_dateils_CNN, "CNN"+ time.strftime("%Y%m%d-%H%M%S")   )
    save_history(train_dateils_Resnt, "ResNet" + time.strftime("%Y%m%d-%H%M%S") )
    print("Historia została zapisana.")
    print("Koniec programu.")

    # save the model
    if not os.path.exists(data_path.model_storage):
        os.makedirs(data_path.model_storage, exist_ok=True)
    
    modelCNNName = data_path.model_storage + "/modelCNN" + time.strftime("%Y%m%d-%H%M%S") + ".pth"
    modelResNetName = data_path.model_storage + "/modelResNet" + time.strftime("%Y%m%d-%H%M%S") + ".pth"
    torch.save(modelResNet, data_path.model_storage + modelResNetName )
    torch.save(modelCNN, data_path.model_storage + modelResNetName)

    upload_model(modelCNNName)
    upload_model(modelResNetName)
    print("Model został zapisany.")

    #
    
    # print(os.listdir(data_path.model_storage))
    # # upload the model
    # models = os.listdir(data_path.model_storage)
    # root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # absolute_models_path = [os.path.join(root_path, data_path.model_storage, model) for model in models]
    # for model in absolute_models_path:
    #     print(f"Wgrywanie modelu {model} na serwer.")
    #     dataset_loader.upload_model(model, model.split("/")[-1])
    # print("Model został wgrany na serwer.")