import torch.optim as optim
from datetime import datetime
from models_schema.params import params
def prepare_model(device, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs).to(device)
    return model

def train_model(model, device, train_loader, criterion, optimizer, epochs):
    training_details = {
        "total_time": None,
        "device": str(device),
        "model_settings": {
            "batch_size": train_loader.batch_size,
            "img_height": params.img_height,
            "img_width": params.img_width,
            "epochs": epochs,
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr']
        },
        "epochs": []
    }

    start_time = datetime.now()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}%")

        training_details["epochs"].append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_acc
        })

    end_time = datetime.now()
    total_time = end_time - start_time
    training_details["total_time"] = str(total_time)

    return training_details
