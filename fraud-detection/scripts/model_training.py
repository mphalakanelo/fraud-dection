import os
import joblib
from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define file paths
file_path = "C:/Users/Kanelo/fraud-dection/fraud-detection/data/creditcard.csv"
model_path = "C:/Users/Kanelo/fraud-dection/fraud-detection/models/fraud_detection_model.pkl"

# Ensure the models directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(file_path)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")
