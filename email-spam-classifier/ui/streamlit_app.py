import streamlit as st
import joblib
import re
import os
import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="Multilingual Email Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .spam-result {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .not-spam-result {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #4caf50 0%, #ffc107 50%, #f44336 100%);
        position: relative;
        margin: 1rem 0;
    }
    .confidence-indicator {
        position: absolute;
        top: -5px;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: white;
        border: 3px solid #2196f3;
        transform: translateX(-50%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .sample-text {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
model = None
model_loaded = False

def clean_text(text: str) -> str:
    """Minimal text cleaning - lowercase and handle unicode safely"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def load_local_model():
    """Load the pretrained spam classifier model locally"""
    global model, model_loaded
    
    try:
        model_path = "../model/spam_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}")
            return False
        
        with st.spinner("Loading model..."):
            model = joblib.load(model_path)
            model_loaded = True
        return True
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
        return False

def predict_with_local_model(text: str) -> Dict[str, Any]:
    """Make prediction using locally loaded model"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    try:
        cleaned_text = clean_text(text)
        prediction = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]
        
        confidence = max(probabilities)
        spam_probability = probabilities[1]
        label = "spam" if prediction == 1 else "not_spam"
        
        return {
            "label": label,
            "confidence": confidence,
            "spam_probability": spam_probability
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def predict_with_api(text: str, api_url: str) -> Dict[str, Any]:
    """Make prediction using FastAPI service"""
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"text": text},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code} - {response.text}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"API connection failed: {str(e)}"}

def get_sample_emails() -> Dict[str, list]:
    """Get sample emails for demonstration"""
    return {
        "English Spam": [
            "You have won a free lottery prize worth $1000",
            "Congratulations! You've been selected for a special offer",
            "Click here to claim your free gift now",
            "Limited time offer! Get 50% off on all products"
        ],
        "English Ham": [
            "Meeting scheduled for tomorrow at 10 AM",
            "Please review the attached document",
            "Your order has been shipped successfully",
            "Thank you for your recent purchase"
        ],
        "Hindi Spam": [
            "आपने एक मुफ्त इनाम जीता है",
            "बधाई हो! आपको विशेष ऑफर के लिए चुना गया है",
            "यहाँ क्लिक करें और अपना मुफ्त उपहार दावा करें",
            "सीमित समय ऑफर! सभी उत्पादों पर 50% छूट पाएं"
        ],
        "Hindi Ham": [
            "कल सुबह 10 बजे की बैठक निर्धारित है",
            "कृपया संलग्न दस्तावेज़ की समीक्षा करें",
            "आपका ऑर्डर सफलतापूर्वक भेज दिया गया है",
            "आपकी हाल की खरीदारी के लिए धन्यवाद"
        ],
        "French Spam": [
            "Vous avez gagné un prix de loterie gratuit",
            "Félicitations! Vous avez été sélectionné pour une offre spéciale",
            "Cliquez ici pour réclamer votre cadeau gratuit",
            "Offre à durée limitée! Obtenez 50% de réduction"
        ],
        "French Ham": [
            "Réunion prévue pour demain à 10h00",
            "Veuillez examiner le document ci-joint",
            "Votre commande a été expédiée avec succès",
            "Merci pour votre achat récent"
        ],
        "Spanish Spam": [
            "¡Has ganado un premio de lotería gratuito",
            "¡Felicidades! Has sido seleccionado para una oferta especial",
            "Haz clic aquí para reclamar tu regalo gratuito",
            "¡Oferta por tiempo limitado! Obtén 50% de descuento"
        ],
        "Spanish Ham": [
            "Reunión programada para mañana a las 10:00 AM",
            "Por favor revisa el documento adjunto",
            "Tu pedido ha sido enviado exitosamente",
            "Gracias por tu compra reciente"
        ]
    }

def main():
    # Header
    st.markdown('<div class="main-header">📧 Multilingual Email Spam Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect spam emails in English, Hindi, French, and Spanish</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model source selection
    model_source = st.sidebar.radio(
        "Model Source:",
        ["Local Model", "API Service"],
        help="Choose whether to use the local model or connect to an API service"
    )
    
    # API configuration (if API is selected)
    api_url = None
    if model_source == "API Service":
        api_url = st.sidebar.text_input(
            "API URL:",
            value="http://localhost:8000",
            help="URL of the FastAPI spam classification service"
        )
        
        # Test API connection
        if st.sidebar.button("Test API Connection"):
            if api_url:
                try:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    if response.status_code == 200:
                        st.sidebar.success("✅ API connection successful!")
                    else:
                        st.sidebar.error(f"❌ API error: {response.status_code}")
                except:
                    st.sidebar.error("❌ Failed to connect to API")
            else:
                st.sidebar.error("Please enter an API URL")
    
    # Load model (if local is selected)
    if model_source == "Local Model":
        if not model_loaded:
            load_local_model()
    
    # Main content area
    st.header("🔍 Classify Email")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Type text", "Select sample"])
    
    email_text = ""
    
    if input_method == "Type text":
        email_text = st.text_area(
            "Enter email text:",
            height=150,
            placeholder="Paste or type the email content here..."
        )
    else:
        # Sample email selection
        samples = get_sample_emails()
        category = st.selectbox("Select category:", list(samples.keys()))
        
        if category:
            sample_emails = samples[category]
            selected_sample = st.selectbox("Select sample email:", sample_emails)
            email_text = st.text_area(
                "Sample email text:",
                value=selected_sample,
                height=150,
                disabled=True
            )
    
    # Classification button
    if st.button("🚀 Classify Email", type="primary", disabled=not email_text.strip()):
        if model_source == "Local Model" and not model_loaded:
            st.error("Model not loaded. Please check the model file.")
            return
        
        if model_source == "API Service" and not api_url:
            st.error("Please enter an API URL.")
            return
        
        # Show spinner during prediction
        with st.spinner("Analyzing email..."):
            time.sleep(0.5)  # Add a small delay for better UX
            
            if model_source == "Local Model":
                result = predict_with_local_model(email_text)
            else:
                result = predict_with_api(email_text, api_url)
        
        # Display results
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # Result display
            label = result["label"]
            confidence = result["confidence"]
            spam_prob = result["spam_probability"]
            
            # Determine styling based on result
            if label == "spam":
                result_class = "spam-result"
                emoji = "🚫"
                label_text = "SPAM"
            else:
                result_class = "not-spam-result"
                emoji = "✅"
                label_text = "NOT SPAM"
            
            # Display result box
            st.markdown(f"""
            <div class="result-box {result_class}">
                <div style="font-size: 2rem;">{emoji}</div>
                <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                    {label_text}
                </div>
                <div style="font-size: 1rem;">
                    Confidence: {confidence:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence visualization
            st.subheader("📊 Confidence Analysis")
            
            # Confidence bar
            confidence_percentage = confidence * 100
            indicator_position = confidence_percentage
            
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-indicator" style="left: {indicator_position}%;"></div>
            </div>
            <p style="text-align: center; margin-top: -1rem;">
                <strong>{confidence_percentage:.1f}%</strong> confidence
            </p>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "🚫 Spam Probability",
                    f"{spam_prob:.3f}",
                    delta=None
                )
            with col2:
                st.metric(
                    "✅ Not Spam Probability",
                    f"{1-spam_prob:.3f}",
                    delta=None
                )
            
            # Additional info
            with st.expander("📋 Detailed Information"):
                st.json({
                    "label": label,
                    "confidence": confidence,
                    "spam_probability": spam_prob,
                    "not_spam_probability": 1 - spam_prob,
                    "model_source": model_source,
                    "text_length": len(email_text),
                    "word_count": len(email_text.split())
                })
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with ❤️ using Naive Bayes + TF-IDF | Supports English, Hindi, French, Spanish
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
