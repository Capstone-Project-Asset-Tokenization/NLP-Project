from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer():
    return TfidfVectorizer()


def extract_features(corpus, vectorizer):
    corpus = corpus.fillna("")
    return vectorizer.fit_transform(corpus)
