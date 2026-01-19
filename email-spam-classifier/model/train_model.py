import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
import joblib
import re

def clean_text(text):
    """Minimal text cleaning - lowercase and handle unicode safely"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def create_sample_dataset():
    """Create a sample dataset for training"""
    spam_samples = [
        "You have won a free lottery prize worth $1000",
        "Congratulations! You've been selected for a special offer",
        "Click here to claim your free gift now",
        "Limited time offer! Get 50% off on all products",
        "You are the lucky winner of our monthly draw",
        "Free money! Transfer to your account immediately",
        "Urgent: Your account needs verification",
        "Special promotion just for you - act fast",
        "Congratulations! You've won a free iPhone",
        "Claim your free vacation package today",
        "You've been selected for exclusive membership",
        "Free trial - no credit card required",
        "Win big prizes in our online lottery",
        "Special discount offer expires soon",
        "Congratulations! You're our millionth visitor",
        "Free gift waiting for you - claim now",
        "You have been chosen for a cash prize",
        "Limited slots available - register now",
        "Exclusive offer just for premium members",
        "You've won a shopping spree worth $5000"
    ]
    
    ham_samples = [
        "Meeting scheduled for tomorrow at 10 AM",
        "Please review the attached document",
        "Your order has been shipped successfully",
        "Thank you for your recent purchase",
        "Can we reschedule our meeting for next week?",
        "The quarterly report is ready for review",
        "Project deadline has been extended by one week",
        "Please find attached the invoice for last month",
        "Team lunch this Friday at 12:30 PM",
        "Your subscription will renew automatically",
        "Weekly team sync meeting notes",
        "Budget approval for Q4 has been granted",
        "Please confirm your attendance for the conference",
        "The server maintenance is scheduled for Sunday",
        "Your flight booking confirmation details",
        "Monthly newsletter - latest updates",
        "Project milestone achieved successfully",
        "Please review and provide feedback on the proposal",
        "Team building event next month",
        "Your package has been delivered successfully"
    ]
    
    data = []
    for text in spam_samples:
        data.append({"text": clean_text(text), "label": 1})
    for text in ham_samples:
        data.append({"text": clean_text(text), "label": 0})
    
    return pd.DataFrame(data)

def main():
    print("🎯 Training Multilingual Email Spam Classifier")
    print("=" * 50)
    
    # Create or load dataset
    print("📊 Creating sample dataset...")
    df = create_sample_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Spam samples: {df['label'].sum()}")
    print(f"Ham samples: {len(df) - df['label'].sum()}")
    
    # Split data
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create pipeline with TF-IDF and Naive Bayes
    print("\n🔧 Building ML pipeline...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=5000,
            lowercase=False  # We handle lowercase in clean_text
        )),
        ('classifier', MultinomialNB(alpha=1.0))
    ])
    
    # Train model
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on training data
    train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred)
    
    # Evaluate on test data
    test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    
    print(f"\n📈 Training Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    
    print(f"\n📋 Classification Report (Test):")
    print(classification_report(y_test, test_pred, target_names=['Ham', 'Spam']))
    
    # Save the model
    model_path = "spam_model.pkl"
    print(f"\n💾 Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("✅ Model saved successfully!")
    
    # Test with sample predictions
    print(f"\n🧪 Sample Predictions:")
    test_samples = [
        "You have won a free lottery prize",
        "Meeting scheduled for tomorrow",
        "Click here to claim your free gift",
        "Please review the attached document"
    ]
    
    for sample in test_samples:
        cleaned_sample = clean_text(sample)
        prediction = pipeline.predict([cleaned_sample])[0]
        probability = pipeline.predict_proba([cleaned_sample])[0]
        confidence = max(probability)
        
        label = "SPAM" if prediction == 1 else "NOT_SPAM"
        print(f"Text: {sample}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.3f}")
        print("-" * 30)

if __name__ == "__main__":
    main()
