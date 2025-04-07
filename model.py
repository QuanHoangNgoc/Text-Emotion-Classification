import torch
import torch.nn as nn

import config

single_config = config.Config()


class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)        # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)       # (batch_size, embedding_dim, seq_len)
        x = self.conv1d(x)           # (batch_size, out_channels, seq_len)
        x = self.relu(x)
        x = self.global_avg_pool(x)  # (batch_size, out_channels, 1)
        x = x.squeeze(2)             # (batch_size, out_channels)
        x = self.fc(x)               # (batch_size, num_classes)
        return x


def create_model(tokenizer):
    model = EmotionClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=single_config.EMBEDDING_DIM,
        num_classes=single_config.NUM_CLASSES
    )
    model.to(single_config.DEVICE)
    return model
