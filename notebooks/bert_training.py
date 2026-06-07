"""
FairHire AI - BERT Resume Screener Training (MODIFIED)
Uses DistilBERT (faster than BERT, same accuracy)
Run this on Google Colab for GPU acceleration

MODIFICATIONS:
- Added bias detection rules
- Added comprehensive test candidates
- Fixed prediction function for local use
- Added model comparison with Random Forest
"""

# ============================================================
# SECTION 1: INSTALL & IMPORT
# ============================================================

# Run this first in Colab:
# !pip install transformers torch pandas scikit-learn

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import json
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

print("✅ All imports successful")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================

CONFIG = {
    # Model settings
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,
    'num_labels': 2,

    # Training settings
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_steps': 100,
    'weight_decay': 0.01,

    # Data settings
    'test_size': 0.2,
    'random_seed': 42,

    # Output settings
    'model_save_path': 'fairhire_bert_model',
    'metrics_save_path': 'model_metrics.json',
    'rf_model_path': 'models/bert_screener/model.pkl'  # For comparison
}

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Set random seeds
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['random_seed'])

# ============================================================
# SECTION 3: BIAS DETECTION RULES
# ============================================================

def detect_bias(resume_text, job_title, prediction, probability):
    """
    Rule-based bias detection layer
    Returns bias flags and adjusted recommendation
    """
    resume_lower = resume_text.lower()
    bias_flags = []

    # Gender bias check
    female_names = ["fatima", "sarah", "emily", "priya", "maria", "laraib", 
                    "aisha", "maryam", "zara", "sana", "hiba", "noor"]
    male_names = ["ahmed", "michael", "john", "david", "muhammad", "ali"]

    has_female_name = any(name in resume_lower for name in female_names)
    has_male_name = any(name in resume_lower for name in male_names)

    if has_female_name and prediction == 0 and probability < 0.6:
        bias_flags.append("potential_gender_bias")

    # Career gap check
    gap_keywords = ["career break", "maternity leave", "paternity leave", 
                    "family care", "health recovery", "sabbatical", 
                    "gap year", "time off", "personal leave"]
    if any(gap in resume_lower for gap in gap_keywords):
        if prediction == 0:
            bias_flags.append("potential_career_gap_bias")

    # Education bias check
    non_prestige = ["community college", "online university", "online degree",
                    "self-taught", "bootcamp", "diploma", "local institute",
                    "state college", "bahria university", "case university"]
    prestige = ["mit", "stanford", "harvard", "berkeley", "cmu", "caltech"]

    has_non_prestige = any(school in resume_lower for school in non_prestige)
    has_prestige = any(school in resume_lower for school in prestige)

    if has_non_prestige and not has_prestige and prediction == 0:
        bias_flags.append("potential_education_bias")

    # Age bias check
    age_indicators = ["20+ years experience", "30+ years", "senior professional",
                      "recent graduate", "fresh graduate", "entry level",
                      "career changer", "mid-career"]
    if any(age in resume_lower for age in age_indicators):
        if prediction == 0:
            bias_flags.append("potential_age_bias")

    # Nationality/Location bias
    countries = ["pakistan", "india", "nigeria", "bangladesh", "philippines",
                 "egypt", "kenya", "vietnam"]
    if any(country in resume_lower for country in countries):
        if prediction == 0 and probability < 0.6:
            bias_flags.append("potential_nationality_bias")

    # Title mismatch bias
    if "no direct" in resume_lower or "not exactly" in resume_lower:
        if prediction == 0:
            bias_flags.append("potential_title_mismatch_bias")

    return bias_flags

def get_bias_recommendation(bias_flags, original_prediction, probability):
    """Generate recommendation based on bias detection"""
    if not bias_flags:
        return {
            'decision': "HIRED ✅" if original_prediction == 1 else "REJECTED ❌",
            'confidence': probability,
            'bias_detected': False,
            'recommendation': 'Standard screening result'
        }

    # If bias detected and candidate was rejected, flag for review
    if original_prediction == 0:
        return {
            'decision': "FLAGGED FOR REVIEW ⚠️",
            'confidence': probability,
            'bias_detected': True,
            'bias_types': bias_flags,
            'recommendation': (
                f"Candidate rejected but potential bias detected: "
                f"{', '.join(bias_flags)}. "
                f"Recommend manual review by hiring manager."
            )
        }

    return {
        'decision': "HIRED ✅",
        'confidence': probability,
        'bias_detected': True,
        'bias_types': bias_flags,
        'recommendation': (
            f"Hired despite potential bias flags: {', '.join(bias_flags)}. "
            f"Verify decision is based on qualifications."
        )
    }

# ============================================================
# SECTION 4: LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """Load and prepare training data"""

    print(f"\n📂 Loading dataset from {file_path}...")

    df = pd.read_csv(file_path)

    print(f"✅ Total samples: {len(df)}")
    print(f"✅ Hired: {df['hired'].sum()} "
          f"({df['hired'].mean()*100:.1f}%)")
    print(f"✅ Rejected: {(df['hired']==0).sum()} "
          f"({(df['hired']==0).mean()*100:.1f}%)")

    # Check for bias examples
    bias_count = df['bias_detected'].sum() if 'bias_detected' in df.columns else 0
    print(f"✅ Bias examples: {bias_count}")

    # Combine resume + job description as input
    df['input_text'] = (
        "RESUME: " + 
        df['resume_text'].fillna('').astype(str) + 
        " [SEP] JOB: " + 
        df['job_description'].fillna('').astype(str)
    )

    # Truncate to reasonable length
    df['input_text'] = df['input_text'].str[:512]

    return df

# ============================================================
# SECTION 5: DATASET CLASS
# ============================================================

class ResumeDataset(Dataset):
    """PyTorch Dataset for resume screening"""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# SECTION 6: TRAINING FUNCTION
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {correct/total*100:.1f}%")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

# ============================================================
# SECTION 7: EVALUATION FUNCTION
# ============================================================

def evaluate(model, dataloader, device):
    """Evaluate model performance"""

    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }

# ============================================================
# SECTION 8: MAIN TRAINING
# ============================================================

def train_fairhire_model(data_path):
    """Complete training pipeline"""

    print("\n" + "="*50)
    print("🚀 FAIRHIRE AI - BERT TRAINING")
    print("="*50)

    # Load data
    df = load_dataset(data_path)

    # Split data
    print("\n📊 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['input_text'].values,
        df['hired'].values,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_seed'],
        stratify=df['hired'].values
    )

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Testing samples: {len(X_test)}")

    # Load tokenizer
    print("\n📥 Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])
    print("✅ Tokenizer loaded")

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = ResumeDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, CONFIG['max_length'])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0 if device.type == 'cuda' else 2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0 if device.type == 'cuda' else 2
    )

    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Testing batches: {len(test_loader)}")

    # Load model
    print("\n📥 Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_labels']
    )
    model.to(device)
    print("✅ Model loaded and moved to device")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Setup scheduler
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )

    # Training loop
    print("\n🎯 Starting training...")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("-"*50)

    best_accuracy = 0
    history = []

    for epoch in range(CONFIG['epochs']):
        print(f"\n📈 EPOCH {epoch+1}/{CONFIG['epochs']}")
        print("-"*30)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        test_metrics = evaluate(model, test_loader, device)

        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_accuracy': round(train_acc, 4),
            'test_loss': round(test_metrics['loss'], 4),
            'test_accuracy': round(test_metrics['accuracy'], 4),
            'test_precision': round(test_metrics['precision'], 4),
            'test_recall': round(test_metrics['recall'], 4),
            'test_f1': round(test_metrics['f1'], 4)
        }
        history.append(epoch_results)

        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  Test Precision: {test_metrics['precision']*100:.2f}%")
        print(f"  Test Recall: {test_metrics['recall']*100:.2f}%")
        print(f"  Test F1: {test_metrics['f1']*100:.2f}%")

        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            print(f"\n💾 Saving best model (accuracy: {best_accuracy*100:.2f}%)...")
            model.save_pretrained(CONFIG['model_save_path'])
            tokenizer.save_pretrained(CONFIG['model_save_path'])
            print("✅ Best model saved!")

    # Final evaluation
    print("\n" + "="*50)
    print("📊 FINAL EVALUATION")
    print("="*50)

    final_metrics = evaluate(model, test_loader, device)

    print(f"\nFinal Test Results:")
    print(f"  Accuracy:  {final_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {final_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {final_metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {final_metrics['f1']*100:.2f}%")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(final_metrics['labels'], final_metrics['predictions'])
    print(f"  True Negatives (Correctly Rejected): {cm[0][0]}")
    print(f"  False Positives (Wrongly Hired):     {cm[0][1]}")
    print(f"  False Negatives (Wrongly Rejected):  {cm[1][0]}")
    print(f"  True Positives (Correctly Hired):    {cm[1][1]}")

    print("\nDetailed Classification Report:")
    print(classification_report(
        final_metrics['labels'],
        final_metrics['predictions'],
        target_names=['Rejected', 'Hired']
    ))

    # Save metrics
    metrics = {
        'config': CONFIG,
        'training_history': history,
        'final_metrics': {
            'accuracy': round(final_metrics['accuracy'], 4),
            'precision': round(final_metrics['precision'], 4),
            'recall': round(final_metrics['recall'], 4),
            'f1': round(final_metrics['f1'], 4)
        },
        'confusion_matrix': cm.tolist(),
        'best_accuracy': round(best_accuracy, 4),
        'training_samples': len(X_train),
        'testing_samples': len(X_test),
        'device': str(device),
        'model_name': CONFIG['model_name']
    }

    with open(CONFIG['metrics_save_path'], 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Metrics saved to: {CONFIG['metrics_save_path']}")
    print(f"✅ Model saved to: {CONFIG['model_save_path']}/")

    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)
    print(f"🏆 Best Accuracy: {best_accuracy*100:.2f}%")

    return model, tokenizer, metrics

# ============================================================
# SECTION 9: PREDICTION WITH BIAS DETECTION
# ============================================================

def predict_resume(resume_text, job_description, model, tokenizer):
    """
    Predict if a candidate should be hired
    Includes bias detection layer
    """

    model.eval()

    # Prepare input
    input_text = (f"RESUME: {resume_text[:300]} "
                  f"[SEP] JOB: {job_description[:200]}")

    # Tokenize
    encoding = tokenizer(
        input_text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )

    probabilities = torch.softmax(outputs.logits, dim=1)
    hire_probability = probabilities[0][1].item()
    prediction = 1 if hire_probability > 0.5 else 0

    # Bias detection
    bias_flags = detect_bias(resume_text, job_description, prediction, hire_probability)
    recommendation = get_bias_recommendation(bias_flags, prediction, hire_probability)

    return {
        'hire_probability': round(hire_probability, 3),
        'raw_decision': "HIRED ✅" if prediction == 1 else "REJECTED ❌",
        'bias_detected': len(bias_flags) > 0,
        'bias_flags': bias_flags,
        'final_decision': recommendation['decision'],
        'confidence': round(recommendation['confidence'] * 100, 1),
        'recommendation': recommendation['recommendation']
    }

# ============================================================
# SECTION 10: COMPREHENSIVE TEST CANDIDATES
# ============================================================

def test_all_candidates(model, tokenizer):
    """Test model with diverse candidates including bias cases"""

    print("\n" + "="*60)
    print("🧪 COMPREHENSIVE CANDIDATE TESTING")
    print("="*60)

    candidates = [
        {
            'name': 'Laraib Kaleem (You - Qualified)',
            'resume': """Laraib Kaleem - AI/ML Engineer and NLP Researcher
                MS Data Science, Bahria University, Pakistan
                BS Computer Science, CASE University
                Published NLP researcher - Springer Nature 2024
                RUATS: Abstractive Text Summarization for Roman Urdu
                AI/ML Trainee at Samsung Innovation Campus 2026
                BI Analyst Intern at United Bank Limited 2024
                Skills: Python, BERT, T5, Transformers, PyTorch,
                TensorFlow, Scikit-learn, SHAP, LIME, Explainable AI,
                NLP, Machine Learning, Deep Learning, Streamlit""",
            'job': 'Senior ML Engineer at Google'
        },
        {
            'name': 'Fatima Khan (Gender Bias Case)',
            'resume': """Fatima Khan Senior Data Scientist
                PhD Computer Science MIT
                8 years experience machine learning
                Published 12 papers top ML conferences
                Expert NLP deep learning Python TensorFlow
                Led teams 10 engineers Amazon
                Skills: Python, TensorFlow, PyTorch, NLP,
                BERT, transformers, model deployment, data analysis""",
            'job': 'Senior ML Engineer at Google'
        },
        {
            'name': 'Sarah Johnson (Career Gap Bias)',
            'resume': """Sarah Johnson Machine Learning Engineer
                MS Computer Science Stanford University
                5 years ML experience before career break
                2 year career break maternity leave
                Recently completed ML refresher fastai Coursera
                Strong Python TensorFlow scikit-learn skills
                Published 3 ML research papers conferences""",
            'job': 'Senior ML Engineer at Google'
        },
        {
            'name': 'Ahmed Raza (Education/Nationality Bias)',
            'resume': """Ahmed Raza Data Analyst
                BS Information Technology Bahria University Pakistan
                Self-taught machine learning Coursera fastai
                3 years experience analyzing large datasets
                Built ML models ecommerce prediction 87 accuracy
                Proficient Python SQL scikit-learn pandas numpy""",
            'job': 'Data Scientist at Microsoft'
        },
        {
            'name': 'Random Person (Unqualified)',
            'resume': """High school diploma
                worked grocery store cashier 2 years
                customer service retail sales
                no technical skills or computer experience
                looking for any job available""",
            'job': 'Senior ML Engineer at Google'
        },
        {
            'name': 'Michael Chen (Qualified - No Bias)',
            'resume': """Michael Chen Senior ML Engineer
                PhD Computer Science Stanford
                10 years Google, Meta, OpenAI
                Led team of 20 ML engineers
                Published 25 papers NeurIPS, ICML
                Expert Python, PyTorch, JAX, transformers
                Built production LLM systems serving 100M users""",
            'job': 'Senior ML Engineer at Google'
        },
        {
            'name': 'Maria Garcia (Age Bias - Senior)',
            'resume': """Maria Garcia Principal Engineer
                20+ years experience in software and ML
                MS Computer Science from State College
                Built first ML pipeline at company in 2010
                Expert in traditional ML and modern deep learning
                Mentored 50+ junior engineers""",
            'job': 'Senior ML Engineer at Google'
        },
        {
            'name': 'David Kim (Title Mismatch Bias)',
            'resume': """David Kim Analytics Professional
                MS Statistics University Chicago
                5 years financial analytics JPMorgan
                Expert Python R SQL statistical modeling
                Built predictive models risk assessment
                No direct Data Scientist title previous roles""",
            'job': 'Data Scientist at Microsoft'
        }
    ]

    results = []
    for candidate in candidates:
        result = predict_resume(candidate['resume'], candidate['job'], model, tokenizer)

        print(f"\n{'='*60}")
        print(f"👤 {candidate['name']}")
        print(f"📋 Job: {candidate['job']}")
        print(f"\n📊 BERT Prediction:")
        print(f"   Raw Decision: {result['raw_decision']}")
        print(f"   Probability: {result['hire_probability']*100:.1f}%")
        print(f"\n🔍 Bias Detection:")
        print(f"   Bias Detected: {'YES ⚠️' if result['bias_detected'] else 'NO ✅'}")
        if result['bias_detected']:
            print(f"   Flags: {', '.join(result['bias_flags'])}")
        print(f"\n✅ FINAL DECISION: {result['final_decision']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Recommendation: {result['recommendation']}")

        results.append({
            'name': candidate['name'],
            **result
        })

    return results

# ============================================================
# SECTION 11: COMPARE WITH RANDOM FOREST
# ============================================================

def compare_with_rf(bert_results, rf_model_path=None):
    """Compare BERT results with Random Forest if available"""

    print("\n" + "="*60)
    print("📊 BERT vs RANDOM FOREST COMPARISON")
    print("="*60)

    # Random Forest results from previous run (hardcoded for comparison)
    rf_results = {
        'Laraib Kaleem (You - Qualified)': {'decision': 'HIRED ✅', 'prob': 52.3},
        'Fatima Khan (Gender Bias Case)': {'decision': 'REJECTED ❌', 'prob': 48.9},
        'Sarah Johnson (Career Gap Bias)': {'decision': 'REJECTED ❌', 'prob': 45.0},
        'Ahmed Raza (Education/Nationality Bias)': {'decision': 'REJECTED ❌', 'prob': 40.1},
        'Random Person (Unqualified)': {'decision': 'HIRED ✅', 'prob': 52.1},
        'Michael Chen (Qualified - No Bias)': {'decision': 'HIRED ✅', 'prob': 85.0},
        'Maria Garcia (Age Bias - Senior)': {'decision': 'REJECTED ❌', 'prob': 42.0},
        'David Kim (Title Mismatch Bias)': {'decision': 'REJECTED ❌', 'prob': 38.0}
    }

    print(f"\n{'Candidate':<45} {'RF Decision':<15} {'BERT Decision':<20} {'Improvement'}")
    print("-"*90)

    for r in bert_results:
        name = r['name']
        bert_dec = r['final_decision']
        bert_prob = r['hire_probability'] * 100

        if name in rf_results:
            rf_dec = rf_results[name]['decision']
            rf_prob = rf_results[name]['prob']

            # Determine if BERT improved
            improved = ""
            if "Unqualified" in name and "REJECTED" in bert_dec:
                improved = "✅ FIXED"
            elif "Bias" in name and "REVIEW" in bert_dec:
                improved = "✅ FLAGGED"
            elif "Qualified" in name and "HIRED" in bert_dec:
                improved = "✅ CORRECT"
            else:
                improved = "⚠️ SAME"

            print(f"{name:<45} {rf_dec:<15} {bert_dec:<20} {improved}")

    print("\n✅ BERT should show better context understanding")
    print("⚠️ Bias cases should be FLAGGED FOR REVIEW, not auto-rejected")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # Path to training data
    # For Google Colab: upload training_data.csv and use:
    DATA_PATH = 'data/resumes_processed/training_data.csv'

    # Alternative for Colab:
    # DATA_PATH = '/content/training_data.csv'

    print("="*60)
    print("🚀 FAIRHIRE AI - BERT TRAINING + BIAS DETECTION")
    print("="*60)
    print(f"📁 Data path: {DATA_PATH}")
    print(f"💾 Model will be saved to: {CONFIG['model_save_path']}/")
    print(f"📊 Metrics will be saved to: {CONFIG['metrics_save_path']}")
    print("="*60)

    # Train model
    model, tokenizer, metrics = train_fairhire_model(DATA_PATH)

    # Test with comprehensive candidates
    test_results = test_all_candidates(model, tokenizer)

    # Compare with Random Forest
    compare_with_rf(test_results)

    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETE!")
    print("="*60)
    print("📥 Download from Colab:")
    print("   1. fairhire_bert_model/ (folder)")
    print("   2. model_metrics.json")
    print("\n📌 Next Steps:")
    print("   - Copy model folder to: models/bert_screener/")
    print("   - Update your app to use BERT + bias detection")
    print("   - Deploy FairHire AI!")

# """
# FairHire AI - BERT Resume Screener Training
# Uses DistilBERT (faster than BERT, same accuracy)
# Run this on Google Colab for GPU acceleration
# """

# # ============================================================
# # SECTION 1: INSTALL & IMPORT
# # ============================================================

# # Run this first in Colab:
# # !pip install transformers torch pandas scikit-learn

# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# # from transformers import (
# #     DistilBertTokenizer,
# #     DistilBertForSequenceClassification,
# #     AdamW,
# #     get_linear_schedule_with_warmup
# # )

# from transformers import (
#     DistilBertTokenizer,
#     DistilBertForSequenceClassification,
#     get_linear_schedule_with_warmup
# )
# from torch.optim import AdamW

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report
# )
# import json
# import os
# import warnings
# warnings.filterwarnings('ignore')

# print("✅ All imports successful")
# print(f"✅ PyTorch version: {torch.__version__}")
# print(f"✅ GPU available: {torch.cuda.is_available()}")

# # ============================================================
# # SECTION 2: CONFIGURATION
# # ============================================================

# CONFIG = {
#     # Model settings
#     'model_name': 'distilbert-base-uncased',
#     'max_length': 256,
#     'num_labels': 2,
    
#     # Training settings
#     'batch_size': 16,
#     'epochs': 3,
#     'learning_rate': 2e-5,
#     'warmup_steps': 100,
#     'weight_decay': 0.01,
    
#     # Data settings
#     'test_size': 0.2,
#     'random_seed': 42,
    
#     # Output settings
#     'model_save_path': 'fairhire_bert_model',
#     'metrics_save_path': 'model_metrics.json'
# }

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() 
#                       else 'cpu')
# print(f"✅ Using device: {device}")

# # Set random seeds
# torch.manual_seed(CONFIG['random_seed'])
# np.random.seed(CONFIG['random_seed'])

# # ============================================================
# # SECTION 3: LOAD DATASET
# # ============================================================

# def load_dataset(file_path):
#     """Load and prepare training data"""
    
#     print(f"\n📂 Loading dataset from {file_path}...")
    
#     df = pd.read_csv(file_path)
    
#     print(f"✅ Total samples: {len(df)}")
#     print(f"✅ Hired: {df['hired'].sum()} "
#           f"({df['hired'].mean()*100:.1f}%)")
#     print(f"✅ Rejected: {(df['hired']==0).sum()} "
#           f"({(df['hired']==0).mean()*100:.1f}%)")
    
#     # Combine resume + job description as input
#     df['input_text'] = (
#         "RESUME: " + 
#         df['resume_text'].fillna('').astype(str) + 
#         " [SEP] JOB: " + 
#         df['job_description'].fillna('').astype(str)
#     )
    
#     # Truncate to reasonable length
#     df['input_text'] = df['input_text'].str[:512]
    
#     return df

# # ============================================================
# # SECTION 4: DATASET CLASS
# # ============================================================

# class ResumeDataset(Dataset):
#     """PyTorch Dataset for resume screening"""
    
#     def __init__(self, texts, labels, tokenizer, max_length):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
        
#         # Tokenize
#         encoding = self.tokenizer(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'label': torch.tensor(label, dtype=torch.long)
#         }

# # ============================================================
# # SECTION 5: TRAINING FUNCTION
# # ============================================================

# def train_epoch(model, dataloader, optimizer, 
#                 scheduler, device):
#     """Train for one epoch"""
    
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     for batch_idx, batch in enumerate(dataloader):
        
#         # Move to device
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
        
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
        
#         loss = outputs.loss
#         logits = outputs.logits
        
#         # Backward pass
#         loss.backward()
        
#         # Clip gradients
#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(), 1.0
#         )
        
#         # Update
#         optimizer.step()
#         scheduler.step()
        
#         # Track metrics
#         total_loss += loss.item()
#         predictions = torch.argmax(logits, dim=1)
#         correct += (predictions == labels).sum().item()
#         total += labels.size(0)
        
#         # Print progress every 50 batches
#         if batch_idx % 50 == 0:
#             print(f"  Batch {batch_idx}/{len(dataloader)} "
#                   f"Loss: {loss.item():.4f}")
    
#     avg_loss = total_loss / len(dataloader)
#     accuracy = correct / total
    
#     return avg_loss, accuracy

# # ============================================================
# # SECTION 6: EVALUATION FUNCTION
# # ============================================================

# def evaluate(model, dataloader, device):
#     """Evaluate model performance"""
    
#     model.eval()
#     all_predictions = []
#     all_labels = []
#     total_loss = 0
    
#     with torch.no_grad():
#         for batch in dataloader:
            
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)
            
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
            
#             loss = outputs.loss
#             logits = outputs.logits
            
#             total_loss += loss.item()
#             predictions = torch.argmax(logits, dim=1)
            
#             all_predictions.extend(
#                 predictions.cpu().numpy()
#             )
#             all_labels.extend(
#                 labels.cpu().numpy()
#             )
    
#     avg_loss = total_loss / len(dataloader)
#     accuracy = accuracy_score(all_labels, all_predictions)
#     precision = precision_score(
#         all_labels, all_predictions, 
#         zero_division=0
#     )
#     recall = recall_score(
#         all_labels, all_predictions,
#         zero_division=0
#     )
#     f1 = f1_score(
#         all_labels, all_predictions,
#         zero_division=0
#     )
    
#     return {
#         'loss': avg_loss,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'predictions': all_predictions,
#         'labels': all_labels
#     }

# # ============================================================
# # SECTION 7: MAIN TRAINING
# # ============================================================

# def train_fairhire_model(data_path):
#     """Complete training pipeline"""
    
#     print("\n" + "="*50)
#     print("🚀 FAIRHIRE AI - BERT TRAINING")
#     print("="*50)
    
#     # Load data
#     df = load_dataset(data_path)
    
#     # Split data
#     print("\n📊 Splitting dataset...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['input_text'].values,
#         df['hired'].values,
#         test_size=CONFIG['test_size'],
#         random_state=CONFIG['random_seed'],
#         stratify=df['hired'].values
#     )
    
#     print(f"✅ Training samples: {len(X_train)}")
#     print(f"✅ Testing samples: {len(X_test)}")
    
#     # Load tokenizer
#     print("\n📥 Loading DistilBERT tokenizer...")
#     tokenizer = DistilBertTokenizer.from_pretrained(
#         CONFIG['model_name']
#     )
#     print("✅ Tokenizer loaded")
    
#     # Create datasets
#     print("\n📦 Creating datasets...")
#     train_dataset = ResumeDataset(
#         X_train, y_train, tokenizer, CONFIG['max_length']
#     )
#     test_dataset = ResumeDataset(
#         X_test, y_test, tokenizer, CONFIG['max_length']
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=CONFIG['batch_size'],
#         shuffle=True,
#         num_workers=2
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=CONFIG['batch_size'],
#         shuffle=False,
#         num_workers=2
#     )
    
#     print(f"✅ Training batches: {len(train_loader)}")
#     print(f"✅ Testing batches: {len(test_loader)}")
    
#     # Load model
#     print("\n📥 Loading DistilBERT model...")
#     model = DistilBertForSequenceClassification.from_pretrained(
#         CONFIG['model_name'],
#         num_labels=CONFIG['num_labels']
#     )
#     model.to(device)
#     print("✅ Model loaded and moved to device")
    
#     # Count parameters
#     total_params = sum(
#         p.numel() for p in model.parameters()
#     )
#     trainable_params = sum(
#         p.numel() for p in model.parameters() 
#         if p.requires_grad
#     )
#     print(f"✅ Total parameters: {total_params:,}")
#     print(f"✅ Trainable parameters: {trainable_params:,}")
    
#     # Setup optimizer
#     optimizer = AdamW(
#         model.parameters(),
#         lr=CONFIG['learning_rate'],
#         weight_decay=CONFIG['weight_decay']
#     )
    
#     # Setup scheduler
#     total_steps = len(train_loader) * CONFIG['epochs']
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=CONFIG['warmup_steps'],
#         num_training_steps=total_steps
#     )
    
#     # Training loop
#     print("\n🎯 Starting training...")
#     print(f"Epochs: {CONFIG['epochs']}")
#     print(f"Batch size: {CONFIG['batch_size']}")
#     print(f"Learning rate: {CONFIG['learning_rate']}")
#     print("-"*50)
    
#     best_accuracy = 0
#     history = []
    
#     for epoch in range(CONFIG['epochs']):
#         print(f"\n📈 EPOCH {epoch+1}/{CONFIG['epochs']}")
#         print("-"*30)
        
#         # Train
#         train_loss, train_acc = train_epoch(
#             model, train_loader, 
#             optimizer, scheduler, device
#         )
        
#         # Evaluate
#         test_metrics = evaluate(model, test_loader, device)
        
#         # Save history
#         epoch_results = {
#             'epoch': epoch + 1,
#             'train_loss': round(train_loss, 4),
#             'train_accuracy': round(train_acc, 4),
#             'test_loss': round(test_metrics['loss'], 4),
#             'test_accuracy': round(test_metrics['accuracy'], 4),
#             'test_precision': round(test_metrics['precision'], 4),
#             'test_recall': round(test_metrics['recall'], 4),
#             'test_f1': round(test_metrics['f1'], 4)
#         }
#         history.append(epoch_results)
        
#         print(f"\n📊 Epoch {epoch+1} Results:")
#         print(f"  Train Loss: {train_loss:.4f}")
#         print(f"  Train Accuracy: {train_acc*100:.2f}%")
#         print(f"  Test Loss: {test_metrics['loss']:.4f}")
#         print(f"  Test Accuracy: "
#               f"{test_metrics['accuracy']*100:.2f}%")
#         print(f"  Test Precision: "
#               f"{test_metrics['precision']*100:.2f}%")
#         print(f"  Test Recall: "
#               f"{test_metrics['recall']*100:.2f}%")
#         print(f"  Test F1: "
#               f"{test_metrics['f1']*100:.2f}%")
        
#         # Save best model
#         if test_metrics['accuracy'] > best_accuracy:
#             best_accuracy = test_metrics['accuracy']
#             print(f"\n💾 Saving best model "
#                   f"(accuracy: {best_accuracy*100:.2f}%)...")
#             model.save_pretrained(CONFIG['model_save_path'])
#             tokenizer.save_pretrained(CONFIG['model_save_path'])
#             print("✅ Best model saved!")
    
#     # Final evaluation
#     print("\n" + "="*50)
#     print("📊 FINAL EVALUATION")
#     print("="*50)
    
#     final_metrics = evaluate(model, test_loader, device)
    
#     print(f"\nFinal Test Results:")
#     print(f"  Accuracy:  {final_metrics['accuracy']*100:.2f}%")
#     print(f"  Precision: {final_metrics['precision']*100:.2f}%")
#     print(f"  Recall:    {final_metrics['recall']*100:.2f}%")
#     print(f"  F1 Score:  {final_metrics['f1']*100:.2f}%")
    
#     # Detailed classification report
#     print("\nDetailed Classification Report:")
#     print(classification_report(
#         final_metrics['labels'],
#         final_metrics['predictions'],
#         target_names=['Rejected', 'Hired']
#     ))
    
#     # Save metrics
#     metrics = {
#         'config': CONFIG,
#         'training_history': history,
#         'final_metrics': {
#             'accuracy': round(final_metrics['accuracy'], 4),
#             'precision': round(final_metrics['precision'], 4),
#             'recall': round(final_metrics['recall'], 4),
#             'f1': round(final_metrics['f1'], 4)
#         },
#         'best_accuracy': round(best_accuracy, 4),
#         'training_samples': len(X_train),
#         'testing_samples': len(X_test),
#         'device': str(device),
#         'model_name': CONFIG['model_name']
#     }
    
#     with open(CONFIG['metrics_save_path'], 'w') as f:
#         json.dump(metrics, f, indent=2)
    
#     print(f"\n✅ Metrics saved to: "
#           f"{CONFIG['metrics_save_path']}")
#     print(f"✅ Model saved to: "
#           f"{CONFIG['model_save_path']}/")
    
#     print("\n" + "="*50)
#     print("✅ TRAINING COMPLETE!")
#     print("="*50)
#     print(f"🏆 Best Accuracy: {best_accuracy*100:.2f}%")
#     print("📌 Next: Download model and push to GitHub")
    
#     return model, tokenizer, metrics

# # ============================================================
# # SECTION 8: PREDICTION FUNCTION
# # ============================================================

# def predict_resume(resume_text, job_description, 
#                    model, tokenizer):
#     """
#     Predict if a candidate should be hired
#     Returns: score (0-1) and decision
#     """
    
#     model.eval()
    
#     # Prepare input
#     input_text = (f"RESUME: {resume_text[:300]} "
#                   f"[SEP] JOB: {job_description[:200]}")
    
#     # Tokenize
#     encoding = tokenizer(
#         input_text,
#         add_special_tokens=True,
#         max_length=256,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
    
#     # Predict
#     with torch.no_grad():
#         outputs = model(
#             input_ids=encoding['input_ids'].to(device),
#             attention_mask=encoding['attention_mask'].to(device)
#         )
    
#     probabilities = torch.softmax(outputs.logits, dim=1)
#     hire_probability = probabilities[0][1].item()
#     decision = "HIRED ✅" if hire_probability > 0.5 else "REJECTED ❌"
    
#     return {
#         'hire_probability': round(hire_probability, 3),
#         'decision': decision,
#         'confidence': round(
#             max(hire_probability, 1-hire_probability), 3
#         )
#     }

# # ============================================================
# # SECTION 9: TEST WITH YOUR RESUME
# # ============================================================

# def test_with_laraib_resume(model, tokenizer):
#     """Test the model with Laraib's actual resume"""
    
#     print("\n" + "="*50)
#     print("🧪 TESTING WITH LARAIB'S RESUME")
#     print("="*50)
    
#     resume = """
#     Laraib Kaleem - AI/ML Engineer and NLP Researcher
#     MS Data Science, Bahria University, Pakistan
#     BS Computer Science, CASE University
    
#     Published NLP researcher - Springer Nature 2024
#     RUATS: Abstractive Text Summarization for Roman Urdu
    
#     Experience:
#     - AI/ML Trainee at Samsung Innovation Campus 2026
#     - BI Analyst Intern at United Bank Limited 2024
    
#     Skills: Python, BERT, T5, Transformers, PyTorch,
#     TensorFlow, Scikit-learn, SHAP, LIME, Explainable AI,
#     NLP, Machine Learning, Deep Learning, Streamlit,
#     Multi-agent AI Systems, Agentic Workflows
    
#     Projects: AutoBiz AI (supply chain), Clinexa AI
#     (healthcare), RUATS NLP System
    
#     Research: Published paper NLP Roman Urdu
#     Springer Nature 2024
#     """
    
#     job = """
#     Senior ML Engineer at Google
#     Requirements: Python, machine learning, NLP,
#     BERT, transformers, PyTorch, TensorFlow,
#     experience with model deployment, research background
#     """
    
#     result = predict_resume(resume, job, model, tokenizer)
    
#     print(f"\nCandidate: Laraib Kaleem")
#     print(f"Job: Senior ML Engineer at Google")
#     print(f"\nResult:")
#     print(f"  Decision: {result['decision']}")
#     print(f"  Hire Probability: {result['hire_probability']*100:.1f}%")
#     print(f"  Confidence: {result['confidence']*100:.1f}%")
    
#     return result

# # ============================================================
# # MAIN EXECUTION
# # ============================================================

# if __name__ == "__main__":
    
#     # For Google Colab: Upload training_data.csv first
#     # Then update this path
#     DATA_PATH = 'data/resumes_processed/training_data.csv'
    
#     # Train model
#     model, tokenizer, metrics = train_fairhire_model(DATA_PATH)
    
#     # Test with your resume
#     test_result = test_with_laraib_resume(model, tokenizer)
    
#     print("\n🎉 Day 5 Complete!")
#     print("📥 Download these files from Colab:")
#     print("  1. fairhire_bert_model/ (folder)")
#     print("  2. model_metrics.json")
#     print("  Then put in: models/bert_screener/")