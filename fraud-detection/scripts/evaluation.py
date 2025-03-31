import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_data

def load_model(filename='fraud_model.pkl'):
    return joblib.load(filename)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    print("Model Evaluation Metrics:", metrics)
    return metrics

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def tune_hyperparameters(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    model = LogisticRegression()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    model = load_model()
    evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, model.predict(X_test))
    best_model = tune_hyperparameters(X_train, y_train)
    joblib.dump(best_model, 'optimized_fraud_model.pkl')
    print("Optimized model saved as optimized_fraud_model.pkl")
