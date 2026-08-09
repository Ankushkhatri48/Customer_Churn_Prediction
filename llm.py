import os
from groq import Groq
import streamlit as st

def get_groq_client():
    """Retrieves Groq API client using streamlit secrets or env variables."""
    # Try streamlit secrets first
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    # Fallback to environment variable
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
        
    if not api_key:
        return None
        
    return Groq(api_key=api_key)

def generate_plain_english_explanation(shap_df, prediction_prob, contract_val, tenure_val):
    """
    Constructs a deterministic prompt with top SHAP features and requests
    a plain-English summary from Groq.
    """
    client = get_groq_client()
    if not client:
        return (
            "⚠️ Groq API Key is not set. Please add your `GROQ_API_KEY` to the environment variables "
            "or streamlit secrets to enable plain-English explanations."
        )
        
    # Get top 5 factors
    top_factors = shap_df.head(5)
    factors_summary = []
    for _, row in top_factors.iterrows():
        # Human-readable format
        factors_summary.append(
            f"- {row['Feature']} (Value: {row['Value']}): {row['Direction']} with contribution score {row['SHAP_Value']:.4f} "
            f"({row['Contribution_Pct']:.1f}% of total absolute impact)"
        )
        
    factors_text = "\n".join(factors_summary)
    
    system_prompt = (
        "You are an expert customer success analyst. Your task is to translate complex machine learning model "
        "explanations (SHAP values) into brief, natural, plain-English summaries that a non-technical manager "
        "can easily understand. Do not invent details or statistics that are not provided. Be concise, direct, "
        "and clear."
    )
    
    user_prompt = f"""
Here is the SHAP explanation for a customer's churn prediction:
- Churn Probability: {prediction_prob:.1f}%
- Tenure: {tenure_val} months
- Contract: {contract_val}

Top 5 contributing factors:
{factors_text}

Provide a short summary (3-4 sentences max) explaining:
1. The customer's churn risk status (High risk if Churn Probability > 50%, otherwise Low/Moderate).
2. The primary reasons pushing this prediction (positive contribution scores increase risk).
3. The factors reducing the risk, if any (negative contribution scores decrease risk).

Ensure you use ONLY the features and directions listed above. Do not mention technical terms like "SHAP values" or "contribution scores" in the final output. Keep it easy to read.
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,  # Low temperature for deterministic behavior
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating LLM explanation: {e}"
