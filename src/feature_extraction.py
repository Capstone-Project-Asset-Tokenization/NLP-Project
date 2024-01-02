from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(corpus):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(corpus)
    return features, vectorizer
