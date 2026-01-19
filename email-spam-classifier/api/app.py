from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import os
import logging
from typing import Dict, Any
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Email Spam Classifier API",
    description="A FastAPI service for classifying emails as spam or not spam across multiple languages",
    version="1.0.0"
)

# Global variables for model
model = None
model_loaded = False

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    spam_probability: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

def clean_text(text: str) -> str:
    """Minimal text cleaning - lowercase and handle unicode safely"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def load_model():
    """Load the pretrained spam classifier model"""
    global model, model_loaded
    
    try:
        model_path = "../model/spam_model.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return False
        
        logger.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        model_loaded = True
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up Multilingual Spam Classifier API...")
    success = load_model()
    if not success:
        logger.warning("Model failed to load. API will return errors for predictions.")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multilingual Email Spam Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Spam classification (POST)",
            "/docs": "Interactive API documentation"
        },
        "model_loaded": model_loaded
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(request: PredictionRequest):
    """Predict if an email text is spam or not"""
    
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text field cannot be empty"
        )
    
    try:
        # Clean input text
        cleaned_text = clean_text(request.text)
        
        # Make prediction
        prediction = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]
        
        # Extract confidence and spam probability
        confidence = max(probabilities)
        spam_probability = probabilities[1]
        
        # Convert prediction to label
        label = "spam" if prediction == 1 else "not_spam"
        
        logger.info(f"Prediction: {label} (confidence: {confidence:.3f})")
        
        return PredictionResponse(
            label=label,
            confidence=round(confidence, 3),
            spam_probability=round(spam_probability, 3)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction for multiple texts"""
    
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if not requests:
        raise HTTPException(
            status_code=400,
            detail="Request list cannot be empty"
        )
    
    try:
        results = []
        
        for req in requests:
            if not req.text or not req.text.strip():
                results.append({
                    "text": req.text,
                    "error": "Text field cannot be empty"
                })
                continue
            
            # Clean input text
            cleaned_text = clean_text(req.text)
            
            # Make prediction
            prediction = model.predict([cleaned_text])[0]
            probabilities = model.predict_proba([cleaned_text])[0]
            
            # Extract confidence and spam probability
            confidence = max(probabilities)
            spam_probability = probabilities[1]
            
            # Convert prediction to label
            label = "spam" if prediction == 1 else "not_spam"
            
            results.append({
                "text": req.text,
                "label": label,
                "confidence": round(confidence, 3),
                "spam_probability": round(spam_probability, 3)
            })
        
        logger.info(f"Batch prediction completed for {len(requests)} texts")
        
        return {
            "results": results,
            "total_processed": len(requests),
            "successful": len([r for r in results if "error" not in r])
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/stats")
async def get_model_stats():
    """Get basic model statistics"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Get model information
        tfidf_vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['classifier']
        
        return {
            "model_type": "Multinomial Naive Bayes",
            "vectorizer": "TF-IDF (character n-grams)",
            "ngram_range": list(tfidf_vectorizer.ngram_range),
            "analyzer": tfidf_vectorizer.analyzer,
            "max_features": tfidf_vectorizer.max_features,
            "feature_count": len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') else None,
            "classifier_alpha": classifier.alpha
        }
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model stats: {str(e)}"
        )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
