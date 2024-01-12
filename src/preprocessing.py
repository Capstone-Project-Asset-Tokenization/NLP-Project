import re
import unicodedata
import nltk.collocations
import io
import os
from nltk.tokenize import word_tokenize
from nltk import BigramCollocationFinder
import pandas as pd

# Class for normalizing text and expanding short forms


class AmharicTextProcessor:
    def __init__(self, expansion_file_dir):
        self.expansion_file_dir = expansion_file_dir
        self.short_form_dict = self._get_short_forms()

    def _get_short_forms(self):
        """Load short form expansions from a file"""
        with open(self.expansion_file_dir, encoding='utf8') as file:
            expansions = {}
            for line in file:
                line = line.strip()
                if line:
                    expanded = line.split("-")
                    expansions[expanded[0].strip()] = expanded[1].replace(
                        " ", '_').strip()
            return expansions

    def expand_short_form(self, word):
        """Expand a short form word to its full form"""
        return self.short_form_dict.get(word, word)

    def normalize_char_level(self, word):
        """Normalize character-level mismatches in Amharic text"""
        # Regular expressions for normalizing similar characters
        replacements = [
            ('[ሃኅኃሐሓኻ]', 'ሀ'), ('[ሑኁዅ]', 'ሁ'), ('[ኂሒኺ]', 'ሂ'),
            ('[ኌሔዄ]', 'ሄ'), ('[ሕኅ]', 'ህ'), ('[ኆሖኾ]', 'ሆ'),
            ('[ሠ]', 'ሰ'), ('[ሡ]', 'ሱ'), ('[ሢ]', 'ሲ'),
            ('[ሣ]', 'ሳ'), ('[ሤ]', 'ሴ'), ('[ሥ]', 'ስ'),
            ('[ሦ]', 'ሶ'), ('[ዓኣዐ]', 'አ'), ('[ዑ]', 'ኡ'),
            ('[ዒ]', 'ኢ'), ('[ዔ]', 'ኤ'), ('[ዕ]', 'እ'),
            ('[ዖ]', 'ኦ'), ('[ጸ]', 'ፀ'), ('[ጹ]', 'ፁ'),
            ('[ጺ]', 'ፂ'), ('[ጻ]', 'ፃ'), ('[ጼ]', 'ፄ'),
            ('[ጽ]', 'ፅ'), ('[ጾ]', 'ፆ'), ('(ሉ[ዋአ])', 'ሏ'),
            ('(ሙ[ዋአ])', 'ሟ'), ('(ቱ[ዋአ])', 'ቷ'), ('(ሩ[ዋአ])', 'ሯ'),
            ('(ሱ[ዋአ])', 'ሷ'), ('(ሹ[ዋአ])', 'ሿ'), ('(ቁ[ዋአ])', 'ቋ'),
            ('(ቡ[ዋአ])', 'ቧ'), ('(ቹ[ዋአ])', 'ቿ'), ('(ሁ[ዋአ])', 'ኋ'),
            ('(ኑ[ዋአ])', 'ኗ'), ('(ኙ[ዋአ])', 'ኟ'), ('(ኩ[ዋአ])', 'ኳ'),
            ('(ዙ[ዋአ])', 'ዟ'), ('(ጉ[ዋአ])', 'ጓ'), ('(ደ[ዋአ])', 'ዷ'),
            ('(ጡ[ዋአ])', 'ጧ'), ('(ጩ[ዋአ])', 'ጯ'), ('(ጹ[ዋአ])', 'ጿ'),
            ('(ፉ[ዋአ])', 'ፏ'), ('[ቊ]', 'ቁ'), ('[ኵ]', 'ኩ')
        ]

        for pattern, replacement in replacements:
            word = re.sub(pattern, replacement, word)
        return word

    def remove_punctuation(self, text):
        """Remove punctuation and special characters from text"""
        return re.sub('[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\+\፡\።\፤\;\፦\፥\፧\፨\፠\፣]', '', text)

    def remove_ascii_and_numbers(self, text):
        """Remove ASCII characters and numbers from text"""
        text = re.sub('[A-Za-z0-9]', '', text)
        return re.sub('[\'\u1369-\u137C\']+', '', text)

    def tokenize(self, corpus):
        """Tokenize a corpus into sentences and words"""
        print('Tokenization ...')
        all_tokens = []
        sentences = re.compile('[!?።(\፡\፡)]+').split(corpus)
        for sentence in sentences:
            tokens = sentence.split()  # assuming non-sentence identifiers are already removed
            all_tokens.extend(tokens)
        return all_tokens

    def collocation_finder(self, tokens, bigram_dir):
        """Find and save frequent bigrams in the given tokens"""
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        # Only consider bigrams that appear more than 3 times
        finder.apply_freq_filter(3)
        # Top 5 bigrams based on chi-square measure
        frequent_bigrams = finder.nbest(bigram_measures.chi_sq, 5)

        with io.open(bigram_dir, "w", encoding="utf8") as file:
            for bigram in frequent_bigrams:
                file.write(bigram[0] + ' ' + bigram[1] + "\n")
        print(frequent_bigrams)

    def normalize_multi_words(self, tokenized_sentence, bigram_dir, corpus):
        """Normalize multi-word expressions in a tokenized sentence"""
        bigrams = set()
        sent_with_bigrams = []
        index = 0
        if not os.path.exists(bigram_dir):
            self.collocation_finder(self.tokenize(corpus), bigram_dir)
            # calling itself recursively
            return self.normalize_multi_words(tokenized_sentence, bigram_dir, corpus)
        else:
            with open(bigram_dir, encoding='utf8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        bigrams.add(line)

            if len(tokenized_sentence) == 1:
                sent_with_bigrams = tokenized_sentence
            else:
                while index <= len(tokenized_sentence) - 2:
                    multi_word = tokenized_sentence[index] + \
                        ' ' + tokenized_sentence[index + 1]
                    if multi_word in bigrams:
                        sent_with_bigrams.append(
                            tokenized_sentence[index] + '' + tokenized_sentence[index + 1])
                        index += 1
                    else:
                        sent_with_bigrams.append(tokenized_sentence[index])
                    index += 1
                    if index == len(tokenized_sentence) - 1:
                        sent_with_bigrams.append(tokenized_sentence[index])

            return sent_with_bigrams

# Additional utility functions


def load_amharic_stop_words(file_path):
    """Load Amharic stop words from a file"""
    with open(file_path, 'r', encoding='utf8') as file:
        return set(file.read().splitlines())


def stop_word_removal(words, stop_words_file):
    """Remove stop words from a list of words"""
    stop_words = load_amharic_stop_words(stop_words_file)
    return [word for word in words if word not in stop_words]


""" Pre-processing section """

# Initialize AmharicTextProcessor with the path to your short forms file and stop words file
pre_processor = AmharicTextProcessor('data/short_form_with_meanings.txt')
stop_words_file = 'data/stop_words.txt'

# Path to your CSV files
input_csv_file = 'data/raw_data/raw_dataset.csv'
output_csv_file = 'data/processed_data/processed_dataset.csv'
bigram_dir = 'data/bigrams.txt'

# Read the CSV file
df = pd.read_csv(input_csv_file)


def preprocess_text(text):
    # Tokenize the text
    print("Original:", text)
    tokens = pre_processor.tokenize(text)

    print("Tokenized:", tokens)

    # Expand short forms in each token
    tokens = [pre_processor.expand_short_form(token) for token in tokens]

    # Normalize each token
    tokens = [pre_processor.normalize_char_level(token) for token in tokens]

    # Remove punctuation and special characters
    tokens = [pre_processor.remove_punctuation(token) for token in tokens]

    # Remove ASCII characters and numbers
    tokens = [pre_processor.remove_ascii_and_numbers(
        token) for token in tokens]

    # Normalize multi-word expressions
    tokens = pre_processor.normalize_multi_words(tokens, bigram_dir, text)

    # Remove stop words
    tokens = stop_word_removal(tokens, stop_words_file)

    # Reconstruct the text
    preprocessed_text = ' '.join(tokens)
    print("Preprocessed:", preprocessed_text)
    return ' '.join(tokens)


# Apply preprocessing to each text entry
df['preprocessed_text'] = df['text'].apply(preprocess_text)

df_preprocessed = df[['preprocessed_text', 'label']]

# Save the preprocessed data to a new CSV file
df_preprocessed.to_csv(output_csv_file, index=False)
