import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
LOCAL_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Columns expected for input/modeling
CATEGORICAL_COLS = [
    'Contract', 'InternetService', 'PaymentMethod', 
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

def generate_synthetic_data(num_rows=1000):
    """Generates a realistic synthetic Telco Churn dataset as a fallback."""
    np.random.seed(42)
    customer_ids = [f"{i:04d}-SYNTH" for i in range(num_rows)]
    
    tenure = np.random.randint(1, 72, size=num_rows)
    monthly_charges = np.random.uniform(18.0, 120.0, size=num_rows)
    
    # Total charges roughly correlates with tenure * monthly_charges
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=num_rows)
    total_charges = np.clip(total_charges, 18.0, None)
    
    # Introduce some random blank strings (approx 1%)
    total_charges_str = []
    for tc in total_charges:
        if np.random.rand() < 0.01:
            total_charges_str.append(" ")
        else:
            total_charges_str.append(f"{tc:.2f}")
            
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_rows, p=[0.55, 0.20, 0.25])
    internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_rows, p=[0.40, 0.45, 0.15])
    payment_methods = np.random.choice([
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ], size=num_rows, p=[0.35, 0.25, 0.20, 0.20])
    
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows, p=[0.30, 0.55, 0.15])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows, p=[0.30, 0.55, 0.15])
    paperless_billing = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.60, 0.40])
    
    # Churn probability based on contract, tenure, tech support
    churn_prob = 0.15
    # High risk for month-to-month, low tenure, no tech support
    churn_prob_array = np.zeros(num_rows)
    for i in range(num_rows):
        prob = 0.1
        if contracts[i] == 'Month-to-month':
            prob += 0.3
        if tenure[i] < 12:
            prob += 0.25
        if tech_support[i] == 'No':
            prob += 0.15
        if internet_services[i] == 'Fiber optic':
            prob += 0.1
        churn_prob_array[i] = np.clip(prob, 0.05, 0.95)
        
    churn = np.where(np.random.rand(num_rows) < churn_prob_array, 'Yes', 'No')
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_str,
        'Contract': contracts,
        'InternetService': internet_services,
        'PaymentMethod': payment_methods,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'PaperlessBilling': paperless_billing,
        'Churn': churn
    })
    return df

def load_data():
    """Loads dataset from URL, local copy, or fallback synthetic data."""
    if os.path.exists(LOCAL_PATH):
        print(f"Loading local dataset: {LOCAL_PATH}")
        df = pd.read_csv(LOCAL_PATH)
    else:
        try:
            print(f"Downloading dataset from {DATA_URL}")
            import urllib.request
            # download with 5 second timeout
            with urllib.request.urlopen(DATA_URL, timeout=5) as response:
                df = pd.read_csv(response)
            df.to_csv(LOCAL_PATH, index=False)
            print("Successfully downloaded dataset.")
        except Exception as e:
            print(f"Failed to download dataset. Generating synthetic data... Error: {e}")
            df = generate_synthetic_data()
            df.to_csv(LOCAL_PATH, index=False)
    return df

def clean_and_prepare(df):
    """Cleans spaces, fills missing values, and prepares target variable."""
    df = df.copy()
    
    # 1. Clean TotalCharges (coerce blank spaces to NaN)
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill TotalCharges with median (or tenure * MonthlyCharges if available)
    median_val = df['TotalCharges'].median()
    if np.isnan(median_val):
        median_val = 0.0
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    
    # 2. Encode Churn label as binary 0 or 1
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def preprocess_features(df, is_training=True, encoders=None):
    """
    Encodes categorical features. Using LabelEncoding/Mapping for simple UI form matching
    while preserving feature semantics.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
        
    # We will use simple integer label encoding for categorical variables.
    # This keeps the feature count low and makes SHAP outputs highly readable.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is not None:
                    # Handle unseen categories by fallback to first category or using transform safely
                    classes = list(le.classes_)
                    df[col] = df[col].apply(lambda x: classes.index(x) if x in classes else 0)
                else:
                    df[col] = 0
                    
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    return df[feature_cols], encoders
