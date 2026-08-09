import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_shap_explainer(model):
    """Initializes and returns a SHAP TreeExplainer."""
    # RandomForestClassifier output shapes: class 0 and class 1. We explain class 1 (Churn).
    explainer = shap.TreeExplainer(model)
    return explainer

def get_global_shap_plot(model, X_train):
    """Generates the SHAP summary plot and returns the figure."""
    explainer = get_shap_explainer(model)
    
    # Use a subset of training data to compute global SHAP values quickly (within 1 second)
    X_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # shap_values is a list for multi-class classification models. Index 1 corresponds to Churn (Yes)
    if isinstance(shap_values, list):
        # Depending on shap version, it might return a list of arrays
        shap_vals_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Some versions return shape (n_samples, n_features, n_classes)
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values
        
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_vals_class1, X_sample, show=False)
    plt.tight_layout()
    return fig

def get_local_shap_explanation(model, X_train, customer_features_df):
    """
    Computes SHAP values for a single customer row.
    Returns:
        - shap_df: DataFrame with feature, value, SHAP value, and percentage contribution.
        - base_value: base value (prior probability) of the explainer.
        - prediction_prob: predicted probability of class 1.
    """
    explainer = get_shap_explainer(model)
    
    # Calculate SHAP values for this specific row
    shap_output = explainer(customer_features_df)
    
    # Extract base value and shap values for class 1 (index 1 if binary classification)
    # TreeExplainer on RandomForestClassifier returns SHAP values in probability scale
    # if model output format is chosen or if we extract class 1
    if len(shap_output.shape) == 3: # (n_samples, n_features, n_classes)
        shap_values_row = shap_output.values[0, :, 1]
        base_val = shap_output.base_values[0, 1]
    elif len(shap_output.shape) == 2:
        # Might already be class 1 or a 2D array
        # Let's inspect shape
        shap_values_row = shap_output.values[0]
        base_val = shap_output.base_values[0]
    else:
        shap_values_row = shap_output.values[0]
        base_val = shap_output.base_values
        
    feature_names = customer_features_df.columns.tolist()
    feature_vals = customer_features_df.iloc[0].values
    
    # Build df
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_vals,
        'SHAP_Value': shap_values_row
    })
    
    # Sort by absolute SHAP value (magnitude of impact)
    shap_df['Abs_SHAP'] = shap_df['SHAP_Value'].abs()
    shap_df = shap_df.sort_values(by='Abs_SHAP', ascending=False).reset_index(drop=True)
    
    # Calculate percentage contribution
    # For visualization, calculate contribution relative to total absolute SHAP sum
    total_abs = shap_df['Abs_SHAP'].sum()
    if total_abs > 0:
        shap_df['Contribution_Pct'] = (shap_df['Abs_SHAP'] / total_abs) * 100
    else:
        shap_df['Contribution_Pct'] = 0.0
        
    # Add direction
    shap_df['Direction'] = np.where(shap_df['SHAP_Value'] > 0, 'Increases Risk', 'Decreases Risk')
    
    return shap_df, base_val
