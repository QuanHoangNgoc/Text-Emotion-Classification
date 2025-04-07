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


def edit_config_file(num_epochs):
    code = open("config.py", "r").read()
    # code = code.replace("MODEL_NAME = 'bert-base-uncased'",
    #                     "MODEL_NAME = 'bert-base-uncased'")
    # code = code.replace("EMBEDDING_DIM = 128", "EMBEDDING_DIM = 128")
    # code = code.replace("BATCH_SIZE = 16", "BATCH_SIZE = 16")
    # code = code.replace("LEARNING_RATE = 0.001", "LEARNING_RATE = 0.001")
    code = code.replace("NUM_EPOCHS = 5", f"NUM_EPOCHS = {num_epochs}")
    # code = code.replace("DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
    #                     "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'")
    # code = code.replace("DATASET_NAME = 'emotion'", "DATASET_NAME = 'emotion'")
    # code = code.replace("EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']",
    #                     "EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']")
    # code = code.replace("NUM_CLASSES = len(EMOTIONS)",
    #                     "NUM_CLASSES = len(EMOTIONS)")
    print(f"Change config file: num_epochs = {num_epochs}")
    open("config.py", "w").write(code)


if __name__ == "__main__":
    edit_config_file(num_epochs=5)
