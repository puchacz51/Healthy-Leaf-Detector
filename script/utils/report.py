import torch
from datetime import datetime
from history import save_history, show_history
import os

models_history_path = os.getenv("MODELS_HISTORY_PATH")  
models_output_path = os.getenv("MODELS_OUTPUT_PATH")


def report_training_results(training_details, model, model_name):
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    filename = f"{models_history_path}/{timestamp}.json"

    save_history(training_details, filename)

    model_save_path = f'{models_output_path}/{model_name}.pth'

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print('Finished Training')

    show_history(filename)
