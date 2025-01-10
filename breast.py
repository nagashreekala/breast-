import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

#  Data Loading and Preprocessing
@st.cache
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

df, data = load_data()

# Display dataset in web app
st.title("Breast Cancer Detection")
st.write("## Dataset Overview")
st.write(df.head())

#  Model Training
@st.cache
def train_model():
    X = df[data.feature_names]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    with open("breast_cancer_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model, accuracy

model, accuracy = train_model()

# model accuracy
st.write("## Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Web App for Prediction
st.write("## Predict Breast Cancer")

#  trained model
with open("breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

#  user inputs for features
user_input = {}
for feature in data.feature_names:
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Predict cancer type
if st.button("Predict"):
    input_data = np.array(list(user_input.values())).reshape(1, -1)
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.write("### Prediction: Malignant")
    else:
        st.write("### Prediction: Benign")

    st.write("### Prediction Confidence")
    st.write(f"Benign: {prediction_proba[0][0] * 100:.2f}%, Malignant: {prediction_proba[0][1] * 100:.2f}%")
