import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import config
from utils import normalize_text

single_config = config.Config()


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
    dataset = load_dataset(single_config.DATASET_NAME)

    # Split dataset
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Normalize texts
    normalized_train_texts = [normalize_text(text) for text in train_texts]
    normalized_test_texts = [normalize_text(text) for text in test_texts]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(single_config.MODEL_NAME)

    def tokenize(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Tokenize data
    train_encodings = tokenize(normalized_train_texts)
    test_encodings = tokenize(normalized_test_texts)

    print("Check dataset: ")
    print(train_encodings[0])
    print(train_labels[0], len(train_labels))
    print(test_encodings[0])
    print(test_labels[0], len(test_labels))

    # Create datasets
    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)
    print("Check dataset: ")
    print(train_dataset[0])
    print(test_dataset[0])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=single_config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=single_config.BATCH_SIZE)

    return train_loader, test_loader, tokenizer


if __name__ == "__main__":
    load_and_preprocess_data()
