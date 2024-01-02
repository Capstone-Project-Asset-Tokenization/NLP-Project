import tkinter as tk
from tkinter import messagebox
import pickle

# Assuming preprocessing.py is in the same directory as gui.py
from .preprocessing import preprocess_text


def detect_hate_speech(model, vectorizer, text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    return model.predict(vectorized_text)[0]


def start_gui():
    root = tk.Tk()
    root.title("Hate Speech Detection")

    # Input field for text
    text_entry = tk.Text(root, height=10, width=50)
    text_entry.pack()
    # Function to handle button click

    def on_check():
        try:
            text = text_entry.get("1.0", "end-1c")
            result = detect_hate_speech(loaded_model, loaded_vectorizer, text)
            result_label.config(
                text="Hate Speech Detected" if result == 1 else "No Hate Speech Detected")
        except Exception as e:
            messagebox.showerror("Error", "An error occurred: " + str(e))
            result_label.config(text="Error in detection.")

    # Button to check for hate speech
    check_button = tk.Button(
        root, text="Check for Hate Speech", command=on_check)
    check_button.pack()

    # Label to display the result
    result_label = tk.Label(root, text="")
    result_label.pack()

    # Load the model and vectorizer
    try:
        with open('models/trained_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        with open('models/vectorizer.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)
    except Exception as e:
        messagebox.showerror(
            "Error", "Failed to load the model or vectorizer: " + str(e))
        loaded_model = None
        loaded_vectorizer = None

    root.mainloop()
