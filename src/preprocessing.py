import re


def preprocess_text(text):
    # Example preprocessing steps
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\([^)]*\)', '', text)  # Remove text in parentheses
    text = re.sub(r'[^\w\sሀ-ፗ]+', '', text)  # Remove non-Amharic characters
    # TODO Add more Amharic-specific preprocessing
    return text
