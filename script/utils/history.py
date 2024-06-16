import json
import matplotlib.pyplot as plt

def save_history(training_details, filename):
    # Zapis danych do pliku JSON
    with open(filename, 'w') as f:
        json.dump(training_details, f, indent=4)

    # Definicja ścieżki do zapisu modelu
    print(f"Training details saved to {filename}")


def show_history(filename, ):
    # Funkcja do ładowania danych treningowych z pliku JSON
    def load_training_details(filename):
        with open(filename, 'r') as f:
            training_details = json.load(f)
        return training_details

    # Funkcja do tworzenia wykresów strat i dokładności
    def plot_training_details(training_details):
        epochs = [epoch_data["epoch"] for epoch_data in training_details["epochs"]]
        losses = [epoch_data["loss"] for epoch_data in training_details["epochs"]]
        accuracies = [epoch_data["accuracy"] for epoch_data in training_details["epochs"]]

        # Wykres strat
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

        # Wykres dokładności
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, label='Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Funkcja do wyświetlania szczegółowych danych treningowych
    def display_training_details(training_details):
        print("Total Training Time:", training_details["total_time"])
        print("Device Used:", training_details["device"])
        print("Model Settings:")
        for key, value in training_details["model_settings"].items():
            print(f"  {key}: {value}")

    # Załaduj dane treningowe
    training_details = load_training_details(filename)

    # Tworzenie wykresów
    plot_training_details(training_details)

    # Wyświetlanie szczegółowych danych treningowych
    display_training_details(training_details)