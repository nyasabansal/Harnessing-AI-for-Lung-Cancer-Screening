import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\User\Downloads\cancer patient data sets.csv", header=0)
    return data
# Preprocess the dataset
def preprocess_data(data):
    # Drop irrelevant columns
    data.drop(['index', 'Patient Id'], axis=1, inplace=True)
    
    # Encode target variable 'Level' using LabelEncoder
    label_encoder_level = LabelEncoder()
    data['Level'] = label_encoder_level.fit_transform(data['Level'])
    
    return data, label_encoder_level
# Train a Random Forest classifier
def train_model(data):
    X = data.drop('Level', axis=1)
    y = data['Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.image(r"C:\Users\User\Downloads\normal-lung-and-lung-cancer-vector.jpg", width=500)
    st.write('Model Accuracy:', accuracy)
    st.subheader('Classification Report')
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        st.text(report)

    
    return model
# Make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction


# Streamlit app
def main():
    st.title('🔬Cancer Level Prediction🔍')
    st.markdown("""<style>
        .big-font {font-size: 24px !important; color: #3366ff;}
        .small-font {font-size: 16px !important; color: #000000;}
        .result-box {padding: 20px; border-radius: 10px; border: 2px solid #3366ff; background-color: #f0f0f5;}
        .report-box {padding: 20px; border-radius: 10px; border: 2px solid #3366ff; background-color: #ffffff;}
    </style>""", unsafe_allow_html=True)
    # Load the dataset
    data = load_data()

    # Preprocess the dataset
    data,label_encoder_level= preprocess_data(data)

    # Train the model
    model = train_model(data)

    # Get user input for prediction
    st.sidebar.title('🛠️ Input')
    age = st.sidebar.slider('Age', min_value=1, max_value=100, value=0)
    gender = st.sidebar.slider('Gender: 1 = Male & 2 =  Female', min_value = 1, max_value = 2)
    air_pollution = st.sidebar.slider('Air Pollution', min_value=0, max_value=8, value=0)
    alcohol_use = st.sidebar.slider('Alcohol use', min_value=0, max_value=8, value=0)
    dust_allergy = st.sidebar.slider('Dust Allergy', min_value=0, max_value=8, value=0)
    occupational_hazards = st.sidebar.slider('OccuPational Hazards', min_value=0, max_value=8, value=0)
    genetic_risk = st.sidebar.slider('Genetic Risk', min_value=0, max_value=8, value=0)
    chronic_lung_disease = st.sidebar.slider('chronic Lung Disease', min_value=0, max_value=8, value=0)
    balanced_diet = st.sidebar.slider('Balanced Diet', min_value=0, max_value=8, value=0)
    obesity = st.sidebar.slider('Obesity', min_value=0, max_value=8, value=0)
    smoking = st.sidebar.slider('Smoking', min_value=0, max_value=8, value=0)
    passive_smoker = st.sidebar.slider('Passive Smoker', min_value=0, max_value=8, value=0)
    chest_pain = st.sidebar.slider('Chest Pain', min_value=0, max_value=9, value=0)
    coughing_of_blood = st.sidebar.slider('Coughing of Blood', min_value=0, max_value=9, value=0)
    fatigue = st.sidebar.slider('Fatigue', min_value=0, max_value=9, value=0)
    weight_loss = st.sidebar.slider('Weight Loss', min_value=0, max_value=9, value=0)
    shortness_of_breath = st.sidebar.slider('Shortness of Breath', min_value=0, max_value=9, value=0)
    wheezing = st.sidebar.slider('Wheezing', min_value=0, max_value=9, value=0)
    swallowing_difficulty = st.sidebar.slider('Swallowing Difficulty', min_value=0, max_value=9, value=0)
    clubbing_of_finger_nails = st.sidebar.slider('Clubbing of Finger Nails', min_value=0, max_value=9, value=0)
    frequent_cold = st.sidebar.slider('Frequent Cold', min_value=0, max_value=7, value=0)
    dry_cough = st.sidebar.slider('Dry Cough', min_value=0, max_value=7, value=0)
    snoring = st.sidebar.slider('Snoring', min_value=0, max_value=7, value=0)

    # Make prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Air Pollution': [air_pollution],
        'Alcohol use': [alcohol_use],
        'Dust Allergy': [dust_allergy],
        'OccuPational Hazards': [occupational_hazards],
        'Genetic Risk': [genetic_risk],
        'chronic Lung Disease': [chronic_lung_disease],
        'Balanced Diet': [balanced_diet],
        'Obesity': [obesity],
        'Smoking': [smoking],
        'Passive Smoker': [passive_smoker],
        'Chest Pain': [chest_pain],
        'Coughing of Blood': [coughing_of_blood],
        'Fatigue': [fatigue],
        'Weight Loss': [weight_loss],
        'Shortness of Breath': [shortness_of_breath],
        'Wheezing': [wheezing],
        'Swallowing Difficulty': [swallowing_difficulty],
        'Clubbing of Finger Nails': [clubbing_of_finger_nails],
        'Frequent Cold': [frequent_cold],
        'Dry Cough': [dry_cough],
        'Snoring': [snoring]
    })
     
    prediction = predict(model, input_data)
    predicted_level = label_encoder_level.inverse_transform(prediction)[0]
    # Display prediction result
    
    st.markdown("---")
    st.subheader('📊Prediction Result')
    
    if prediction[0] == 0:
        st.write('Predicted Level of Lung Cancer: Low')
    elif prediction[0] == 1:
        st.write('Predicted Level of Lung Cancer: Medium')
    else:
        st.write('Predicted Level of Lung Cancer: High')
        
        
    st.balloons()    


    

if __name__ == '__main__':
    main()