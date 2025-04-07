import re
import string
import unicodedata


def normalize_text(text):
    """
    Normalize text by:
    1. Converting to lowercase
    2. Removing URLs
    3. Removing HTML tags
    4. Removing punctuation
    5. Removing extra whitespace
    6. Normalizing Unicode characters
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
