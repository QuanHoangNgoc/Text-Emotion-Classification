import torch


class Config:
    # Model parameters
    MODEL_NAME = "bert-base-uncased"
    EMBEDDING_DIM = 128

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    DATASET_NAME = "emotion"

    # Emotion labels
    EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    NUM_CLASSES = len(EMOTIONS)
