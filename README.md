# Disease-Prediction-Chatbot
Disease-Prediction-Chatbot

     Brief Description :- The Disease Prediction Chatbot is an intelligent conversational agent designed 
     to assist individuals by predicting potential diseases based on their symptoms. The system uses Natural 
     Language Processing  (NLP) and machine learning algorithms to analyze user input (symptoms) and provide 
     possible disease diagnoses. It is intended for use in medical applications to help users identify symptoms 
     and guide  them to seek appropriate healthcare.
                      
Technology/Framework/Tool :-

     Programming Language: Python
     Machine Learning: Scikit-learn (for model building)
     NLP: TF-IDF Vectorization (for processing symptoms)
     Modeling: Logistic Regression (for classification)
     Development Tools: Jupyter Notebooks, PyCharm/VSCode
     Libraries: Pandas, NumPy, Scikit-learn, NLTK
     Data Storage: Text-based file (diseases_data.txt) for storing disease and symptom data
            
My Role :- Machine Learning Developer & Data Preparation As the Machine Learning Developer, my primary 
     responsibility was to prepare the dataset, preprocess the symptoms using text vectorization techniques, 
     and train the classification model using Logistic Regression. I also worked on the chatbot implementation,
     ensuring smooth integration of the  model for real-time prediction.

Roles & Responsibilities :- Team Lead :- Responsible for coordinating the overall project, ensuring that all parts of the
     project come together smoothly, and guiding the team to meet deadlines. 

Machine Learning Developer (My Role) :-
     o Preprocessed and cleaned the symptom data.
     o Implemented TF-IDF vectorization to convert textual data into a machine-readable format.
     o Trained the Logistic Regression model for disease classification.
     o Implemented real-time prediction capabilities for the chatbot.
     o Worked with the team to integrate the model with the user interface (chatbot). 
                  
Backend Developer :-
          o Developed the data-loading module and designed the system's backend for symptom-disease mapping.
          o Ensured that the data storage and retrieval process was optimized for fast query results. 
          
Challenges Encountered :-

Data Preprocessing :- The dataset of symptoms required significant preprocessing, including tokenization, 
     stemming, and handling missing data. Ensuring the symptoms were represented effectively for the model was a challenge.
          
Model Accuracy :- The model’s accuracy was initially low due to the small dataset. We had to experiment with different
     algorithms and preprocessing techniques to improve the model's prediction  accuracy.
          
Text Classification Limitations :- Processing natural language symptoms and mapping them to diseases using TF-IDF 
     had its limitations, especially when the input symptoms were complex or ambiguous. Fine-tuning the vectorization and trying 
     different classifiers (such as Random Forest or Naive Bayes) helped improve results.
       
Real-time Prediction :- Integrating the trained model into the chatbot was challenging because we needed to ensure that the system
     responded in real-time, which required optimizing the model’s prediction time without compromising accuracy. 
          
Situation/Scenario :- The chatbot is designed for non-experts to identify potential diseases based on symptoms. 
     For example, if a   user types "fever, cough, sore throat," the chatbot predicts Flu or COVID-19, providing users with immediate guidance. It is especially useful in regions where access to healthcare professionals may be limited, as it helps individuals assess whether they need to seek medical attention.

Scenario Example:

 User Input: “I have fever, cough, and tiredness.”
 Chatbot Output: “Based on your symptoms, the predicted disease is: Flu. If the symptoms persist,
     please consult a healthcare professional.”
     
Conclusion :-
     The Disease Prediction Chatbot is a powerful tool that combines machine learning and natural language
     processing to provide users with real-time disease predictions based on their symptoms. By leveraging 
     a Logistic Regression model and TF-IDF text vectorization, the system can accurately classify diseases
     even with a basic dataset. Although challenges in data preprocessing, model accuracy, and 
     integration were encountered, the project was a success in terms of its practical application 
     in healthcare.The project demonstrates a useful application of AI in the healthcare domain, and there is 
     potential for future improvements, such as incorporating larger datasets, adding more diseases, and expanding 
     the system to a web or mobile platform for broader accessibility.
     
CODE : [https://github.com/SoaibAslam/Disease-Prediction-Chatbot/commit/d6ad72cc6a4a844f4f46ebceb0776436c0489cb8](https://github.com/SoaibAslam/Disease-Prediction-Chatbot/commit/f9c03314dc5df574b86d3e0b9e091cb703473a57)