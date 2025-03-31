import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocess_data

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics

def save_model(model, filename='fraud_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:", metrics)
    save_model(model)
