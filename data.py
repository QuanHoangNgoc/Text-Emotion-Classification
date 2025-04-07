import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import Config


class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]


def load_and_preprocess_data():
    # Load dataset
    dataset = load_dataset(Config.DATASET_NAME)

    # Split dataset
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    def tokenize(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Tokenize data
    train_encodings = tokenize(train_texts)
    test_encodings = tokenize(test_texts)

    # Create datasets
    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    return train_loader, test_loader, tokenizer
