# -*- coding: utf-8 -*-

import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

base_dir = os.path.dirname(__file__)

# Construct the relative path to the model
model_path = os.path.join(base_dir, '../models/fraud_detection_model.pkl')

# Load the trained model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found! Ensure 'fraud_detection_model.pkl' is in the 'models' folder.")
    st.stop()

# Function to preprocess data
def preprocess_data(df):
    df = df.dropna()  # Drop missing values

    # Scale features using StandardScaler
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    # Split features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Balance dataset using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

#  display confusion matrix and classification report
def display_metrics(y_true, y_pred):
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(cm)

# Streamlit app
def main():
    st.title("Fraud Detection with Machine Learning")

    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display the dataset
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Preprocess the data
        X, y = preprocess_data(df)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display performance metrics
        display_metrics(y_test, y_pred)

if __name__ == "__main__":
    main()
