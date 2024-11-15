import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default file path
DEFAULT_FILE_PATH = r"C:\Users\SOAIB ASLAM\OneDrive\Desktop\Disease Prediction with chatbot\Disease-Prediction-Chatbot\diseases_data.txt"

def load_data(file_path=DEFAULT_FILE_PATH):
    """
    Load disease and symptom data from a text file.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    diseases = []
    symptoms = []

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            disease = None
            symptom_list = []

            for line in lines:
                line = line.strip()

                if line.startswith("Disease:"):
                    if disease:
                        diseases.append(disease)
                        symptoms.append(symptom_list)
                    disease = line.replace("Disease:", "").strip()
                    symptom_list = []
                elif line.startswith("Symptoms:"):
                    symptom_list = line.replace("Symptoms:", "").strip().split(", ")

            # Append the last disease and its symptoms
            if disease:
                diseases.append(disease)
                symptoms.append(symptom_list)

        logging.info(f"Loaded {len(diseases)} diseases from the file.")
        return pd.DataFrame({"Disease": diseases, "Symptoms": symptoms})
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_data(data):
    """
    Prepare the data for machine learning: vectorize symptoms and set up labels.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        symptoms = [" ".join(symptom_list) for symptom_list in data["Symptoms"]]
        X = vectorizer.fit_transform(symptoms)  # Vectorized symptom data
        y = data["Disease"]  # Target labels
        return X, y, vectorizer
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def train_model(X, y):
    """
    Train a logistic regression model for disease prediction.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=2000, solver='liblinear')  # Suitable solver for small datasets
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model Accuracy: {accuracy:.2f}")
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def predict_disease(symptoms_input, vectorizer, model):
    """
    Predict the disease based on user-input symptoms.
    """
    try:
        symptoms_input = symptoms_input.strip().lower()
        X_input = vectorizer.transform([symptoms_input])
        predicted_disease = model.predict(X_input)[0]
        return predicted_disease
    except Exception as e:
        logging.error(f"Error during disease prediction: {e}")
        return "Error in prediction."

def chatbot(vectorizer, model):
    """
    Chatbot interface for disease prediction.
    """
    print("Welcome to the Disease Prediction Chatbot!")
    print("Enter your symptoms separated by commas (e.g., 'fever, cough, tiredness').")
    print("Type 'exit' to quit.")

    while True:
        symptoms_input = input("Your symptoms: ").strip()
        
        if symptoms_input.lower() == 'exit':
            print("ALLAH HAFIZ")
            break
        
        if not symptoms_input:
            print("Please enter some symptoms.")
            continue

        disease = predict_disease(symptoms_input, vectorizer, model)
        print(f"Based on your symptoms, the predicted disease is: {disease}")

if __name__ == "__main__":
    try:
        # Load data
        data = load_data(DEFAULT_FILE_PATH)
        # Prepare data
        X, y, vectorizer = prepare_data(data)
        # Train model
        model = train_model(X, y)
        # Start chatbot
        chatbot(vectorizer, model)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
