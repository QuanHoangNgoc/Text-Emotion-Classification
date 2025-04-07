# Emotion Classification Project

This project classifies text into different emotions using a deep learning model. The model is trained on the Emotion dataset and uses BERT tokenization with a custom CNN architecture.

## Project Structure

- `config.py`: Configuration parameters for the model and training
- `data.py`: Data loading and preprocessing utilities
- `model.py`: Model architecture definition
- `train.py`: Training and evaluation script
- `predict.py`: Script for making predictions
- `requirements.txt`: Project dependencies

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training:

```bash
python train.py
```

3. Make predictions:

```bash
python predict.py
```

## Model Architecture

The model uses:

- BERT tokenization
- Word embeddings
- 1D Convolutional layers
- Global average pooling
- Fully connected layer for classification

## Configuration

You can modify the following parameters in `config.py`:

- Model parameters (embedding dimension, number of classes)
- Training parameters (batch size, learning rate, epochs)
- Device (CPU/GPU)
- Dataset name
- Emotion labels
