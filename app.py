import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import load_model_assets
from explain import get_global_shap_plot, get_local_shap_explanation
from llm import generate_plain_english_explanation

# Page config
st.set_page_config(
    page_title="Telco Churn Explainer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
st.markdown("""
<style>
    /* Premium font and background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        background-color: #fafbfc;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid #eef2f6;
        margin-bottom: 20px;
    }
    
    .metric-val-high {
        font-size: 3rem;
        font-weight: 700;
        color: #e63946;
        line-height: 1;
        margin-bottom: 8px;
    }
    
    .metric-val-low {
        font-size: 3rem;
        font-weight: 700;
        color: #2a9d8f;
        line-height: 1;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Highlight containers */
    .explanation-box {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 20px;
        border-radius: 4px 12px 12px 4px;
        margin-top: 15px;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
    }
</style>
""", unsafe_allow_html=True)

# Load model assets and cache
@st.cache_resource
def get_assets():
    return load_model_assets()

assets = get_assets()
model = assets['model']
encoders = assets['encoders']
metrics = assets['metrics']
feature_cols = assets['feature_cols']
X_train = assets['X_train']

# Sidebar Header & Overall Performance
st.sidebar.markdown("# Churn Portal Admin")
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Performance")
st.sidebar.markdown(f"**Accuracy:** {metrics['accuracy']:.2%}")
st.sidebar.markdown(f"**Precision:** {metrics['precision']:.2%}")
st.sidebar.markdown(f"**Recall (Sensitivity):** {metrics['recall']:.2%}")
st.sidebar.markdown(f"**F1-Score:** {metrics['f1_score']:.2%}")

# Confusion matrix visual in sidebar
st.sidebar.markdown("#### Confusion Matrix")
cm = np.array(metrics['confusion_matrix'])
cm_df = pd.DataFrame(
    cm, 
    index=['Actual Stay', 'Actual Churn'], 
    columns=['Predicted Stay', 'Predicted Churn']
)
st.sidebar.table(cm_df)

# Global SHAP Summary Plot in Sidebar
st.sidebar.subheader("🌍 Global Feature Importance")
@st.cache_data
def load_global_shap_plot():
    return get_global_shap_plot(model, X_train)

fig_global = load_global_shap_plot()
st.sidebar.pyplot(fig_global)

# Title & Description
st.title("🔮 Customer Churn Prediction & Explainability")
st.markdown("Enter customer details below to predict their probability of churning, see the primary driving factors (SHAP), and view a plain-English reason summary.")

# Form Input Layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 📋 Customer Profile Form")
    with st.form("customer_form"):
        # Numeric inputs
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=250.0, value=70.0, step=1.0)
        
        # Categorical inputs
        contract = st.selectbox(
            "Contract Type",
            options=['Month-to-month', 'One year', 'Two year']
        )
        internet_service = st.selectbox(
            "Internet Service",
            options=['Fiber optic', 'DSL', 'No']
        )
        payment_method = st.selectbox(
            "Payment Method",
            options=[
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ]
        )
        online_security = st.selectbox(
            "Online Security Add-on",
            options=['No', 'Yes', 'No internet service']
        )
        tech_support = st.selectbox(
            "Tech Support Add-on",
            options=['No', 'Yes', 'No internet service']
        )
        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=['Yes', 'No']
        )
        
        submit_btn = st.form_submit_button("Predict Churn Risk")

# On Submit Predict and Explain
if submit_btn or 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = True
    
    # Auto-calculate TotalCharges based on tenure * monthly_charges
    total_charges = tenure * monthly_charges
    
    # Build raw input DataFrame
    input_data = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'PaperlessBilling': paperless_billing
    }])
    
    # Process inputs through the encoders saved during training
    # Import copy of preprocess_features that uses existing encoders
    from data_manager import preprocess_features
    X_input, _ = preprocess_features(input_data, is_training=False, encoders=encoders)
    
    # Predict Churn probability
    proba = model.predict_proba(X_input)[0, 1]
    churn_percentage = proba * 100
    
    # Compute Local SHAP explanation
    shap_df, base_val = get_local_shap_explanation(model, X_train, X_input)
    
    with col2:
        st.markdown("### 🔍 Risk Analysis & Explainability")
        
        # Risk gauge/metric card
        risk_class = "metric-val-high" if churn_percentage >= 50 else "metric-val-low"
        risk_text = "HIGH RISK" if churn_percentage >= 50 else "LOW/MODERATE RISK"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Probability ({risk_text})</div>
            <div class="{risk_class}">{churn_percentage:.1f}%</div>
            <div>Base model average risk: {base_val*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Plot Local SHAP Contributions
        st.markdown("#### Primary Drivers Pushing Risk Up or Down")
        
        fig_local, ax_local = plt.subplots(figsize=(8, 4.5))
        
        # We only plot the features that have non-zero SHAP values
        plot_df = shap_df.head(6).copy()
        
        # Color code: Positive SHAP = Red (increases risk), Negative SHAP = Green (decreases risk)
        colors = ['#e63946' if x > 0 else '#2a9d8f' for x in plot_df['SHAP_Value']]
        
        # Horizontal bar chart
        bars = ax_local.barh(plot_df['Feature'], plot_df['SHAP_Value'], color=colors, edgecolor='none', height=0.6)
        
        # Styling the plot
        ax_local.spines['top'].set_visible(False)
        ax_local.spines['right'].set_visible(False)
        ax_local.spines['left'].set_color('#cbd5e1')
        ax_local.spines['bottom'].set_color('#cbd5e1')
        ax_local.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')
        
        # Add labels to the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 if width >= 0 else width - 0.01
            ha_align = 'left' if width >= 0 else 'right'
            ax_local.text(
                label_x_pos, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:+.2f}", 
                va='center', 
                ha=ha_align, 
                fontsize=9, 
                fontweight='semibold',
                color='#1e293b'
            )
            
        ax_local.set_xlabel("Risk Contribution Magnitude (SHAP value)", fontsize=10, color='#475569')
        ax_local.set_title("Impact on Customer's Churn Risk", fontsize=12, fontweight='bold', color='#0f172a', pad=15)
        plt.tight_layout()
        
        st.pyplot(fig_local)
        
        # Plain English Summary from Groq LLM
        st.markdown("#### 💬 Plain-English Risk Summary (Groq LLM)")
        with st.spinner("Analyzing risk factors..."):
            explanation = generate_plain_english_explanation(
                shap_df, churn_percentage, contract, tenure
            )
            
        st.markdown(f"""
        <div class="explanation-box">
            <p style="margin: 0; font-size: 1rem; color: #1e293b; line-height: 1.6;">
                {explanation}
            </p>
        </div>
        """, unsafe_allow_html=True)
