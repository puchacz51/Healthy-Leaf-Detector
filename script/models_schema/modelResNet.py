import torch.nn as nn
from torchvision import  models

class ResNetModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Modify the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


