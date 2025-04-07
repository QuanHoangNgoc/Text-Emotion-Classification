import torch


class Config:
    _instance = None

    def __new__(cls):  # Single Disign Pattern
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Model parameters
        self.MODEL_NAME = "bert-base-uncased"
        self.EMBEDDING_DIM = 128
        self.NUM_CLASSES = 6  # sadness, joy, love, anger, fear, surprise

        # Training parameters
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 5

        # Device
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Dataset
        self.DATASET_NAME = "emotion"

        # Emotion labels
        self.EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def reset(self):
        self._initialize()


if __name__ == "__main__":
    single_config = Config()
    print(single_config.NUM_EPOCHS)
