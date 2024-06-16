import os
import shutil
import random

from utils.dictionary import data_path

def _split_to_healthy_and_disease(original_dir: str, output_dir: str):
    print(f"Przetwarzanie danych z {original_dir} do {output_dir}...")
    # Przechodzenie przez wszystkie podfoldery w katalogu danych
    for folder in os.listdir(original_dir):
        folder_path = os.path.join(original_dir, folder)
        # Sprawdzenie czy ścieżka prowadzi do folderu
        if os.path.isdir(folder_path):
            # Określenie klasy
            cls: str = data_path.processed_healthy if "healthy" in folder.lower() else data_path.processed_disease

            # Przechodzenie przez wszystkie pliki w podfolderze
            for i, file_name in enumerate(os.listdir(folder_path)):
                src: str = os.path.join(folder_path, file_name)                             # Ścieżka do oryginalnego pliku                 
                dst: str = os.path.join(cls, f"{folder}_{i+1}.jpg")             # Ścieżka do docelowego pliku
                print(f"Kopiowanie pliku {src} do {dst}")
                shutil.copy(src, dst)
        else:
            print(f"{folder_path} nie jest folderem")


def _move_files(source_dir, dest_dir, num_samples):
    files = os.listdir(source_dir)
    random.shuffle(files)                               # Losowa permutacja plików
    selected_files = files[:num_samples]                # Wybór n próbek
    for file in selected_files:
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(dest_dir, file)
        shutil.move(source_file, dest_file)             # Przeniesienie pliku


def _move_remaining_files(source_dir, dest_dir):
    files = os.listdir(source_dir)
    for file in files:
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(dest_dir, file)
        shutil.move(source_file, dest_file)             # Przeniesienie pliku

def remove_directories():
    # Usunięcie katalogów jesli istnieją
    shutil.rmtree(data_path.processed_home, ignore_errors=True)
    shutil.rmtree(data_path.processed_healthy, ignore_errors=True)
    shutil.rmtree(data_path.processed_disease, ignore_errors=True)

    shutil.rmtree(data_path.train_healthy, ignore_errors=True)
    shutil.rmtree(data_path.train_disease, ignore_errors=True)

    shutil.rmtree(data_path.val_healthy, ignore_errors=True)
    shutil.rmtree(data_path.val_disease, ignore_errors=True)
    


def create_directories():
    # Utworzenie katalogów
    os.makedirs(data_path.processed_home, exist_ok=True)
    os.makedirs(data_path.processed_healthy, exist_ok=True)
    os.makedirs(data_path.processed_disease, exist_ok=True)

    os.makedirs(data_path.train_healthy, exist_ok=True)
    os.makedirs(data_path.train_disease, exist_ok=True)

    os.makedirs(data_path.val_healthy, exist_ok=True)
    os.makedirs(data_path.val_disease, exist_ok=True)


def split():
    # checj if processed directory exists and not empty

    remove_directories()
    # Utworzenie katalogów
    create_directories()

    print("Rozpoczęto podział danych na zdrowe i chore liście...")
    _split_to_healthy_and_disease(data_path.original, data_path.processed_home)
    print("Zakończono podział danych na zdrowe i chore liście.")

    # Przeniesienie próbek z katalogu healthy do walidacji
    print('Przenoszenie próbek do katalogu walidacyjnego...')
    _move_files(data_path.processed_healthy, data_path.val_healthy, 5000)
    _move_files(data_path.processed_disease, data_path.val_disease, 5000)

    # Przeniesienie reszty danych do katalogu treningowego
    _move_remaining_files(data_path.processed_healthy, data_path.train_healthy)
    _move_remaining_files(data_path.processed_disease, data_path.train_disease)
    print('Przenoszenie próbek do katalogu walidacyjnego zakończone.')