import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer

import config
import control

single_config = config.Config()


def train_model():
    # Load data
    train_loader, test_loader, tokenizer = control.load_and_preprocess_data()

    # Create model
    model = control.create_model(tokenizer)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=single_config.LEARNING_RATE)

    # Training loop
    for epoch in range(single_config.NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch, labels in train_loader:
            batch = {key: val.to(single_config.DEVICE)
                     for key, val in batch.items()}
            labels = labels.to(single_config.DEVICE)

            optimizer.zero_grad()
            outputs = model(batch["input_ids"])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{single_config.NUM_EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch, labels in test_loader:
                batch = {key: val.to(single_config.DEVICE)
                         for key, val in batch.items()}
                labels = labels.to(single_config.DEVICE)

                outputs = model(batch["input_ids"])
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

    return model, tokenizer


# def save_model(model, tokenizer):
#     model.save_pretrained("model")
#     tokenizer.save_pretrained("tokenizer")


# def load_trained_model(model, tokenizer):
#     model = create_model(tokenizer)
#     model.load_state_dict(torch.load("model"))
#     tokenizer = AutoTokenizer.from_pretrained("tokenizer")
#     return model, tokenizer


def get_main_outcome():
    model, tokenizer = train_model()
    return model, tokenizer


if __name__ == "__main__":
    get_main_outcome()
