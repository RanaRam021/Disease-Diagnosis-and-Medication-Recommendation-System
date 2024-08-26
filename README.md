# Disease-Diagnosis-and-Medication-Recommendation-System

# Overview
The Disease Diagnosis and Medication Recommendation System is an advanced web application designed to assist users in diagnosing diseases based on symptoms and providing tailored recommendations for medication, precautions, workouts, and diets. By leveraging the power of data science and machine learning, specifically using a Random Forest Classifier, this system offers accurate and actionable insights to users.

# Features
Symptom Input: Users can input their symptoms to receive a potential diagnosis. 

Disease Prediction: The system utilizes machine learning to predict the disease based on the input symptoms.

Recommendations: Provides personalized recommendations including:

Medications: Suggested medicines for the diagnosed disease.

Precautions: Important precautions to follow.

Workouts: Recommended exercises to aid recovery or manage symptoms.

Diets: Dietary suggestions to support health and wellness.

# Technologies Used
Backend: Flask - A micro web framework for Python.

Frontend: Bootstrap - A CSS framework for building responsive, mobile-first websites.

Machine Learning: Random Forest Classifier - An ensemble learning method used for classification tasks, known for its robustness and accuracy.

Data Science: Data preprocessing, feature selection, and model training were essential parts of the project, showcasing the ability to work with large datasets and extract meaningful insights.

Deployment: The application can be deployed on various platforms, including Azure App Service.

# How It Works
User Input: Users enter their symptoms into the application form.

Prediction: The Flask backend processes the input and utilizes the Random Forest Classifier to predict the disease based on the trained model.

Results Display: The system displays the predicted disease and relevant recommendations in an organized, user-friendly tabbed interface.

# Installation
To set up the project locally:

Clone the Repository:
    
    git clone https://github.com/yourusername/disease-diagnosis-and-medication-recommendation-system.git

Navigate to the Project Directory:

    cd disease-diagnosis-and-medication-recommendation-system

Create and Activate a Virtual Environment:

    python -m venv venv
    source venv\Scripts\activate

Install Dependencies:

    pip install -r requirements.txt

Run the Flask Application:

    python app.py

Open your Browser and go to:

    http://127.0.0.1:5000 to access the application.

# Usage
Enter Symptoms: Input the symptoms in the provided text box and click "Predict".

View Results: Navigate through the tabs to see the disease prediction and associated recommendations.

# Contributing
Contributions are welcome! If you have suggestions for improvements or find issues, please open an issue or submit a pull request.

# Acknowledgments
Bootstrap: For the beautiful and responsive UI.

Flask: For the lightweight and powerful web framework.

Random Forest Classifier: For its powerful classification capabilities.

Data Science & Machine Learning: For providing the foundation of this project, showcasing the integration of AI in healthcare.
