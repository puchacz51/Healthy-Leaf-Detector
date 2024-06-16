from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models_schema.params import params


def get_train_loader(train_data_dir):
    transform = transforms.Compose([
        transforms.Resize((params.img_height, params.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    
    return train_loader

