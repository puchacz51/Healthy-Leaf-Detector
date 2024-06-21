import os

class Model_params():
    def __init__(self):
        self.img_height = os.getenv('IMG_HEIGHT', 128)
        self.img_width = os.getenv('IMG_WIDTH', 128)
        self.epochs = os.getenv('EPOCHS', 10)
        self.batch_size = os.getenv('BATCH_SIZE', 32)
        self.learning_rate = os.getenv('LEARNING_RATE', 0.001)

params = Model_params()