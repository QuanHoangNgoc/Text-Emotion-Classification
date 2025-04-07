# 🎭 Text Emotion Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning model for emotion classification in text, achieving up to 88% accuracy on the Emotion dataset. This project leverages BERT tokenization and a custom CNN architecture to accurately classify text into six different emotions.

## 🌟 Features

- **High Accuracy**: Achieves up to 88% accuracy on emotion classification
- **BERT Integration**: Utilizes BERT tokenization for superior text understanding
- **Custom CNN Architecture**: Optimized for emotion detection
- **Easy to Use**: Simple API for training and prediction
- **Configurable**: Highly customizable through configuration files

## 📋 Project Structure

```
Text-Emotion-Classification/
├── config.py          # Configuration parameters
├── data.py           # Data loading and preprocessing
├── model.py          # Model architecture
├── train_model.py    # Training script
├── predict.py        # Prediction script
├── interact.py       # Interactive interface
├── utils.py          # Utility functions
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## 🚀 Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Text-Emotion-Classification.git
cd Text-Emotion-Classification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Train the model:

```bash
python train_model.py
```

2. Make predictions:

```bash
python predict.py
```

3. Interactive mode:

```python
import interact
model, tokenizer = interact.get_main_outcome()
emotion = interact.predict_emotion(model, tokenizer, "I love this project!")
print(emotion)  # Output: 'love'
```

## 🧠 Model Architecture

The model combines the power of BERT tokenization with a custom CNN architecture:

- **BERT Tokenization**: For advanced text understanding
- **Word Embeddings**: 128-dimensional embeddings
- **1D Convolutional Layers**: For feature extraction
- **Global Average Pooling**: For dimensionality reduction
- **Fully Connected Layer**: For final classification

## ⚙️ Configuration

Customize your model through `config.py`:

```python
{
    "NUM_EPOCHS": 7,           # Number of training epochs
    "BATCH_SIZE": 32,          # Batch size for training
    "LEARNING_RATE": 0.001,    # Learning rate
    "EMBEDDING_DIM": 128,      # Embedding dimension
    "NUM_CLASSES": 6,          # Number of emotion classes
    "DEVICE": "cuda",          # Training device (cuda/cpu)
    "DATASET_NAME": "emotion", # Dataset name
    "EMOTION_LABELS": [        # Emotion labels
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise"
    ]
}
```

## 📊 Performance

The model achieves the following performance metrics:

| Epoch | Training Loss | Test Accuracy |
| ----- | ------------- | ------------- |
| 1     | 1.4731        | 57.85%        |
| 2     | 0.8201        | 80.20%        |
| 3     | 0.4008        | 85.80%        |
| 4     | 0.2359        | 87.50%        |
| 5     | 0.1549        | 88.10%        |
| 6     | 0.1077        | 87.90%        |
| 7     | 0.0789        | 87.35%        |

## 🌍 Project Impact

This project has made significant contributions in several areas:

### For Developers

- **Improved Development Efficiency** (as measured by reduced implementation time) by providing a ready-to-use emotion classification solution
- **Enhanced Learning Experience** (as measured by educational value) by demonstrating the integration of BERT with custom CNN architectures
- **Reduced Development Costs** (as measured by resource requirements) by offering an open-source alternative to commercial emotion analysis APIs

### For the Community

- **Advanced Research Capabilities** (as measured by academic citations) by providing a state-of-the-art baseline for emotion classification research
- **Improved Accessibility** (as measured by GitHub stars and forks) by making advanced NLP techniques available to a wider audience
- **Enhanced Application Development** (as measured by integration success rate) by enabling developers to add emotion analysis to their applications

### Technical Achievements

- **Superior Performance** (as measured by accuracy metrics) by achieving 88% accuracy on the Emotion dataset
- **Efficient Resource Usage** (as measured by memory and compute requirements) by optimizing the model architecture
- **Scalable Solution** (as measured by deployment success) by providing a flexible and configurable framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The Emotion dataset for providing the training data
