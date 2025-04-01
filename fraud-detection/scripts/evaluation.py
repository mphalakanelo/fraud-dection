import joblib
import numpy as np
from preprocessing import preprocess_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define file paths
file_path = "C:/Users/Kanelo/fraud-dection/fraud-detection/data/creditcard.csv"
model_path = "C:/Users/Kanelo/fraud-dection/fraud-detection/models/fraud_detection_model.pkl"

# Load and preprocess data
print("🔄 Preprocessing data...")
X_train, X_test, y_train, y_test = preprocess_data(file_path)
print("✅ Data preprocessing completed!")

# Load the trained model
print("🔄 Loading trained model...")
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model evaluation completed! Accuracy: {accuracy:.4f}")

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
