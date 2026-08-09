import pickle
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from data_manager import load_data, clean_and_prepare, preprocess_features, NUMERIC_COLS, CATEGORICAL_COLS

MODEL_FILE = "churn_model_assets.pkl"

def train_and_save_model():
    """Trains the model, calculates metrics, and saves the model assets."""
    print("Loading data...")
    raw_df = load_data()
    
    print("Cleaning and preparing data...")
    cleaned_df = clean_and_prepare(raw_df)
    
    # Preprocess features
    X_processed, encoders = preprocess_features(cleaned_df, is_training=True)
    y = cleaned_df['Churn']
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10
    )
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Detailed Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(), # Convert to list for JSON/saving
        'classification_report': report
    }
    
    # Print metrics
    print("\n=== MODEL PERFORMANCE METRICS ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=================================\n")
    
    # Store everything we need in one dictionary
    assets = {
        'model': model,
        'encoders': encoders,
        'metrics': metrics,
        'feature_cols': NUMERIC_COLS + CATEGORICAL_COLS,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(assets, f)
        
    print(f"Model and preprocessors successfully saved to {MODEL_FILE}")
    return assets

def load_model_assets():
    """Loads the model assets or trains a new model if they do not exist."""
    if not os.path.exists(MODEL_FILE):
        print(f"{MODEL_FILE} not found. Training model now...")
        return train_and_save_model()
    
    with open(MODEL_FILE, 'rb') as f:
        assets = pickle.load(f)
    return assets

if __name__ == "__main__":
    train_and_save_model()
