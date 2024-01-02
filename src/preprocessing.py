import re
import unicodedata
from nltk.tokenize import word_tokenize
# No need for NLTK's lemmatizer since it doesn't support Amharic


# Function to load Amharic stop words from a text file
def loadAmharicStopWords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return set(stopwords)


def preprocess_text(text):
    # Normalize Unicode characters to ensure consistency
    text = unicodedata.normalize('NFC', text)

    # Remove punctuation (consider Amharic punctuation as well)
    text = re.sub(r'[^\w\s፡።፣፤፥፦፧፨]', '', text)

    # Remove text in parentheses (this might not be applicable for Amharic and can be removed if unnecessary)
    text = re.sub(r'\([^)]*\)', '', text)

    #  Remove non-Amharic characters
    text = re.sub(r'[^\w\sሀ-ፗ]+', '', text)

    # Tokenization - consider using a language-specific tokenizer if available
    words = word_tokenize(text)

    # TODO Stopword removal - modify the stopword list if necessary
    stop_words = loadAmharicStopWords('data/stop_words.txt')
    words = [w for w in words if w not in stop_words]

    # TODO: Lemmatization - implement language-specific lemmatization if available
    # words = [lemmatize(w) for w in words]

    # TODO: Stemming - implement language-specific stemming if available
    # words = [stem(w) for w in words]

    # Join the words back into a string
    text = " ".join(words)

    return text
