import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    """Handles missing values by removing or imputing them."""
    df = df.dropna()
    return df

def normalize_features(df):
    """Scales the numerical features using StandardScaler."""
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    return df

def balance_dataset(X, y):
    """Balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique)."""
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def preprocess_data(file_path):
    """Loads, cleans, normalizes, and balances the dataset."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = normalize_features(df)
    
    # Splitting features and labels
    X = df.drop(columns=['Class'])  
    y = df['Class']
    
    # Balancing the dataset
    X_balanced, y_balanced = balance_dataset(X, y)
    
    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# 🚀 Prevent execution when imported into model_training.py
if __name__ == "__main__":
    file_path = "C:/Users/Kanelo/fraud-dection/fraud-detection/data/creditcard.csv"
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    print("✅ Data preprocessing completed successfully!")
