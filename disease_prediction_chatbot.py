import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path=r"C:\\Users\\SOAIB ASLAM\\OneDrive\\Desktop\\Disease Prediction with chatbot\\Disease-Prediction-Chatbot\\diseases_data.txt"):
    diseases = []
    symptoms = []

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

        diseases.append(disease)
        symptoms.append(symptom_list)

    return pd.DataFrame({"Disease": diseases, "Symptoms": symptoms})

def prepare_data(data):
    vectorizer = TfidfVectorizer(stop_words="english")
    symptoms = [" ".join(symptom_list) for symptom_list in data["Symptoms"]]
    X = vectorizer.fit_transform(symptoms)  
    y = data["Disease"]
    return X, y, vectorizer

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=2000, solver='liblinear')  
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy: {accuracy:.2f}")

        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return model

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def predict_disease(symptoms_input, vectorizer, model):
    try:
        symptoms_input = symptoms_input.strip().lower()  
        X_input = vectorizer.transform([symptoms_input])  
        predicted_disease = model.predict(X_input)[0]
        return predicted_disease
    except Exception as e:
        logging.error(f"Error during disease prediction: {e}")
        return "Error in prediction."

def chatbot(vectorizer, model):
    print("Welcome to the Disease Prediction Chatbot!")
    print("Enter your symptoms separated by commas (e.g., 'fever, cough, tiredness')")
    print("Type 'exit' to quit.")
    
    while True:
        symptoms_input = input("Your symptoms: ") 
        
        if symptoms_input.lower() == 'exit':  
            print("ALLAH HAFIZ")
            break
        
        if symptoms_input.strip() == "":
            print("Please enter some symptoms.")
            continue
        
        disease = predict_disease(symptoms_input, vectorizer, model)
        print(f"Based on your symptoms, the predicted disease is: {disease}")

if __name__ == "__main__":
    try:
        data = load_data("diseases_data.txt")
        X, y, vectorizer = prepare_data(data)

        model = train_model(X, y)

        chatbot(vectorizer, model)

    except Exception as e:
        logging.error(f"Fatal error: {e}")