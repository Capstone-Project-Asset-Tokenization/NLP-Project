from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(X, y):
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Training the model using Support Vector Machine
    model = SVC(kernel='linear')

    model.fit(X_train, y_train)

    # Evaluating the model
    y_pred = model.predict(X_test)
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Drawing confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5,
                square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(
        accuracy_score(y_test, y_pred))
    plt.title(all_sample_title, size=15)
    plt.show()

    return model
