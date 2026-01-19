import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import re
import os

def clean_text(text):
    """Minimal text cleaning - lowercase and handle unicode safely"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def load_model():
    """Load the pretrained spam classifier model"""
    model_path = "../model/spam_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train_model.py first.")
    
    print(f"📦 Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    return model

def load_test_samples():
    """Load multilingual test samples"""
    samples_path = "../data/samples/multilingual_test_samples.csv"
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Test samples not found at {samples_path}")
    
    print(f"📊 Loading test samples from {samples_path}...")
    df = pd.read_csv(samples_path)
    print(f"✅ Loaded {len(df)} test samples")
    return df

def evaluate_multilingual():
    """Evaluate model on multilingual test samples"""
    print("🌐 Multilingual Email Spam Classifier Evaluation")
    print("=" * 60)
    
    # Load model and data
    model = load_model()
    df = load_test_samples()
    
    # Clean text
    print("\n🧹 Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Make predictions
    print("🔮 Making predictions...")
    predictions = model.predict(df['cleaned_text'])
    probabilities = model.predict_proba(df['cleaned_text'])
    
    # Add predictions to dataframe
    df['prediction'] = predictions
    df['confidence'] = np.max(probabilities, axis=1)
    df['spam_probability'] = probabilities[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(df['label'], df['prediction'])
    precision = precision_score(df['label'], df['prediction'])
    
    print(f"\n📈 Overall Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(df['label'], df['prediction'], target_names=['Ham', 'Spam']))
    
    # Confusion Matrix
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(df['label'], df['prediction'])
    print(f"True Negatives: {cm[0,0]} (Ham correctly identified)")
    print(f"False Positives: {cm[0,1]} (Ham incorrectly marked as Spam)")
    print(f"False Negatives: {cm[1,0]} (Spam incorrectly marked as Ham)")
    print(f"True Positives: {cm[1,1]} (Spam correctly identified)")
    
    # Show sample predictions by language
    print(f"\n🌍 Sample Predictions by Language:")
    
    # English samples
    english_samples = df[df['text'].str.contains(r'^[a-zA-Z\s\!\?\.\,\-]+$', regex=True)].head(5)
    if not english_samples.empty:
        print(f"\n🇺🇸 English Samples:")
        for _, row in english_samples.iterrows():
            label = "SPAM" if row['label'] == 1 else "NOT_SPAM"
            pred = "SPAM" if row['prediction'] == 1 else "NOT_SPAM"
            status = "✅" if row['label'] == row['prediction'] else "❌"
            print(f"{status} Text: {row['text'][:50]}...")
            print(f"   Actual: {label} | Predicted: {pred} | Confidence: {row['confidence']:.3f}")
    
    # Hindi samples (Devanagari script)
    hindi_samples = df[df['text'].str.contains(r'[\u0900-\u097F]', regex=True)].head(5)
    if not hindi_samples.empty:
        print(f"\n🇮🇳 Hindi Samples:")
        for _, row in hindi_samples.iterrows():
            label = "SPAM" if row['label'] == 1 else "NOT_SPAM"
            pred = "SPAM" if row['prediction'] == 1 else "NOT_SPAM"
            status = "✅" if row['label'] == row['prediction'] else "❌"
            print(f"{status} Text: {row['text'][:50]}...")
            print(f"   Actual: {label} | Predicted: {pred} | Confidence: {row['confidence']:.3f}")
    
    # French samples
    french_samples = df[df['text'].str.contains(r'[àâäéèêëïîôöùûüÿç]', regex=True)].head(5)
    if not french_samples.empty:
        print(f"\n🇫🇷 French Samples:")
        for _, row in french_samples.iterrows():
            label = "SPAM" if row['label'] == 1 else "NOT_SPAM"
            pred = "SPAM" if row['prediction'] == 1 else "NOT_SPAM"
            status = "✅" if row['label'] == row['prediction'] else "❌"
            print(f"{status} Text: {row['text'][:50]}...")
            print(f"   Actual: {label} | Predicted: {pred} | Confidence: {row['confidence']:.3f}")
    
    # Spanish samples
    spanish_samples = df[df['text'].str.contains(r'[ñáéíóúü]', regex=True)].head(5)
    if not spanish_samples.empty:
        print(f"\n🇪🇸 Spanish Samples:")
        for _, row in spanish_samples.iterrows():
            label = "SPAM" if row['label'] == 1 else "NOT_SPAM"
            pred = "SPAM" if row['prediction'] == 1 else "NOT_SPAM"
            status = "✅" if row['label'] == row['prediction'] else "❌"
            print(f"{status} Text: {row['text'][:50]}...")
            print(f"   Actual: {label} | Predicted: {pred} | Confidence: {row['confidence']:.3f}")
    
    # Error analysis
    errors = df[df['label'] != df['prediction']]
    if not errors.empty:
        print(f"\n❌ Error Analysis ({len(errors)} misclassifications):")
        for _, row in errors.iterrows():
            actual = "SPAM" if row['label'] == 1 else "NOT_SPAM"
            pred = "SPAM" if row['prediction'] == 1 else "NOT_SPAM"
            print(f"Text: {row['text'][:60]}...")
            print(f"   Actual: {actual} | Predicted: {pred} | Confidence: {row['confidence']:.3f}")
            print("-" * 50)
    
    # High confidence predictions
    high_conf = df[df['confidence'] > 0.9].head(5)
    if not high_conf.empty:
        print(f"\n🎯 High Confidence Predictions (>0.9):")
        for _, row in high_conf.iterrows():
            label = "SPAM" if row['label'] == 1 else "NOT_SPAM"
            pred = "SPAM" if row['prediction'] == 1 else "NOT_SPAM"
            status = "✅" if row['label'] == row['prediction'] else "❌"
            print(f"{status} Text: {row['text'][:50]}...")
            print(f"   Actual: {label} | Predicted: {pred} | Confidence: {row['confidence']:.3f}")
    
    return df, accuracy, precision

def interactive_test():
    """Interactive testing with custom input"""
    print(f"\n🎮 Interactive Testing Mode")
    print("Enter text to classify (or 'quit' to exit):")
    
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    while True:
        user_input = input("\n📝 Enter text: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        cleaned_input = clean_text(user_input)
        prediction = model.predict([cleaned_input])[0]
        probability = model.predict_proba([cleaned_input])[0]
        confidence = max(probability)
        spam_prob = probability[1]
        
        label = "SPAM" if prediction == 1 else "NOT_SPAM"
        emoji = "🚫" if prediction == 1 else "✅"
        
        print(f"{emoji} Prediction: {label}")
        print(f"📊 Confidence: {confidence:.3f}")
        print(f"📈 Spam Probability: {spam_prob:.3f}")

if __name__ == "__main__":
    # Run evaluation
    try:
        df, accuracy, precision = evaluate_multilingual()
        
        # Ask if user wants interactive testing
        choice = input(f"\n🎮 Would you like to try interactive testing? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_test()
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Make sure you've trained the model first by running: python model/train_model.py")
