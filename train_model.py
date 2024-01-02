import pandas as pd
from src.feature_extraction import create_vectorizer, extract_features
from src.model_training import train_model
import pickle
from src.preprocessing import preprocess_text


def main():
    # Load the raw dataset
    # assuming csv files.......
    data_path = 'data/raw_data/raw_dataset.csv'
    df = pd.read_csv(data_path)

    # Preprocess your dataset
    df['text'] = df['text'].apply(preprocess_text)

    # Save your processed dataset
    df.to_csv('data/processed_data/processed_dataset.csv', index=False)

    # Load the processed dataset
    # assuming csv files.......
    data_path = 'data/processed_data/processed_dataset.csv'
    df = pd.read_csv(data_path)

    # Assuming 'text' column for features and 'label' column for labels.........
    X = df['text']
    y = df['label']

    # Create and use the vectorizer for feature extraction
    vectorizer = create_vectorizer()
    X_vectorized = extract_features(X, vectorizer)

    # Train the model
    trained_model = train_model(X_vectorized, y)

    # Save the trained model and vectorizer
    with open('models/trained_model.pkl', 'wb') as file:
        pickle.dump(trained_model, file)

    with open('models/vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)


if __name__ == "__main__":
    main()
