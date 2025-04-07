import torch
from transformers import AutoTokenizer

import utils
from config import Config
from control import create_model


def predict_emotion(model, tokenizer, sentence):
    # Normalize the sentence
    normalized_sentence = utils.normalize_text(sentence)

    model.eval()
    encoded = tokenizer([normalized_sentence], padding=True,
                        truncation=True, return_tensors="pt")
    encoded = {key: val.to(Config.DEVICE) for key, val in encoded.items()}

    with torch.no_grad():
        output = model(encoded["input_ids"])
        prediction = torch.argmax(output, dim=1).cpu().item()

    return Config.EMOTIONS[prediction]


if __name__ == "__main__":
    # Load model and tokenizer (you would typically load these from saved checkpoints)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = create_model(tokenizer)

    # Example prediction
    sample_sentence = "I'm so excited about this new adventure!"
    print(f"Original sentence: {sample_sentence}")

    normalized_sentence = utils.normalize_text(sample_sentence)
    print(f"Normalized sentence: {normalized_sentence}")

    predicted_emotion = predict_emotion(model, tokenizer, sample_sentence)
    print(f"Predicted Emotion: {predicted_emotion}")
