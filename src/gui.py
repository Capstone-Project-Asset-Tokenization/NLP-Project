import tkinter as tk
from tkinter import messagebox
import pickle
from .preprocessing import preprocess_text
from .feature_extraction import extract_features


def detect_hate_speech(model, vectorizer, text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    return model.predict(vectorized_text)[0]


def start_gui():
    root = tk.Tk()
    root.title("Hate Speech Detection")

    # Input field
    text_entry = tk.Text(root, height=10, width=50)
    text_entry.pack()

    # Result label
    result_label = tk.Label(root, text="")
    result_label.pack()

    def on_check():
        text = text_entry.get("1.0", "end-1c")
        result = detect_hate_speech(loaded_model, loaded_vectorizer, text)
        if result == 1:
            result_label.config(text="Hate Speech Detected")
        else:
            result_label.config(text="No Hate Speech Detected")

    check_button = tk.Button(
        root, text="Check for Hate Speech", command=on_check)
    check_button.pack()

    # TODO Load model and vectorizer
    with open('models/trained_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    # TODO Assume vectorizer is saved and loaded similarly
    with open('models/vectorizer.pkl', 'rb') as file:
        loaded_vectorizer = pickle.load(file)

    root.mainloop()
