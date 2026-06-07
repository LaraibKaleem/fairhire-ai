"""
FairHire AI - Resume Screener Model
Using Random Forest + TF-IDF (Fast & Accurate)
Trains in seconds on CPU
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline
import shap
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("🚀 FAIRHIRE AI - MODEL TRAINING")
print("="*50)

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    'model_type': 'RandomForest',
    'test_size': 0.2,
    'random_seed': 42,
    'n_estimators': 100,
    'max_depth': 20,
    'max_features': 500,
    'model_save_path': 'models/bert_screener',
    'metrics_save_path': 'models/model_metrics.json'
}

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load training dataset"""
    
    print("\n📂 Loading dataset...")
    
    df = pd.read_csv(
        'data/resumes_processed/training_data.csv'
    )
    
    print(f"✅ Total samples: {len(df)}")
    print(f"✅ Hired: {df['hired'].sum()} "
          f"({df['hired'].mean()*100:.1f}%)")
    print(f"✅ Rejected: {(df['hired']==0).sum()} "
          f"({(df['hired']==0).mean()*100:.1f}%)")
    
    # Combine resume + job as input
    df['input_text'] = (
        df['resume_text'].fillna('').astype(str) +
        " " +
        df['job_title'].fillna('').astype(str) +
        " " +
        df['resume_category'].fillna('').astype(str)
    )
    
    return df

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(df):
    """Train Random Forest classifier"""
    
    print("\n📊 Preparing features...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['input_text'].values,
        df['hired'].values,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_seed'],
        stratify=df['hired'].values
    )
    
    print(f"✅ Train: {len(X_train)} samples")
    print(f"✅ Test: {len(X_test)} samples")
    
    # Create pipeline
    print("\n🔧 Creating model pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=CONFIG['max_features'],
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=CONFIG['n_estimators'],
            max_depth=CONFIG['max_depth'],
            random_state=CONFIG['random_seed'],
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    # Train
    print("\n🎯 Training Random Forest...")
    print("(Should take 10-30 seconds...)")
    
    pipeline.fit(X_train, y_train)
    print("✅ Training complete!")
    
    # Evaluate
    print("\n⏳ Evaluating...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(
            y_test, y_pred, zero_division=0
        ),
        'recall': recall_score(
            y_test, y_pred, zero_division=0
        ),
        'f1': f1_score(
            y_test, y_pred, zero_division=0
        )
    }
    
    print("\n" + "="*50)
    print("📊 RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Rejected', 'Hired']
    ))
    
    return pipeline, X_train, X_test, y_test, metrics

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def setup_shap(pipeline, X_train):
    """Setup SHAP explainer"""
    
    print("\n🔍 Setting up SHAP explainer...")
    
    tfidf = pipeline.named_steps['tfidf']
    rf = pipeline.named_steps['classifier']
    
    X_train_tfidf = tfidf.transform(X_train[:100])
    X_sample = X_train_tfidf[:5].toarray()
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    
    feature_names = tfidf.get_feature_names_out()
    n_features = len(feature_names)
    
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1][0]).flatten()
    else:
        sv = np.array(shap_values[0]).flatten()
    
    sv = sv[:n_features]
    top_indices = np.argsort(np.abs(sv))[-10:]
    
    print("✅ SHAP explainer working!")
    print("\n📊 Top 10 Most Important Features:")
    for idx in reversed(top_indices):
        val = float(sv[idx])
        print(f"  {feature_names[idx]}: {val:.4f}")
    
    return explainer, feature_names

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_candidate(pipeline, resume_text, job_title):
    """
    Predict if candidate should be hired
    Returns probability and decision
    """
    
    input_text = f"{resume_text} {job_title}"
    
    prob = pipeline.predict_proba([input_text])[0][1]
    decision = "HIRED ✅" if prob > 0.5 else "REJECTED ❌"
    
    return {
        'hire_probability': round(prob, 3),
        'hire_percentage': round(prob * 100, 1),
        'decision': decision,
        'confidence': round(
            max(prob, 1-prob) * 100, 1
        )
    }

# ============================================================
# SAVE MODEL
# ============================================================

def save_model(pipeline, metrics, X_train):
    """Save model and metrics"""
    
    os.makedirs(CONFIG['model_save_path'], exist_ok=True)
    
    # Save model
    model_file = os.path.join(
        CONFIG['model_save_path'], 
        'model.pkl'
    )
    with open(model_file, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\n✅ Model saved: {model_file}")
    
    # Save metrics
    metrics_data = {
        'config': CONFIG,
        'final_metrics': {
            'accuracy': round(metrics['accuracy'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'f1': round(metrics['f1'], 4)
        },
        'model_type': 'RandomForest + TF-IDF',
        'training_samples': len(X_train),
        'features': CONFIG['max_features']
    }
    
    with open(CONFIG['metrics_save_path'], 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✅ Metrics saved: {CONFIG['metrics_save_path']}")

# ============================================================
# TEST CANDIDATES
# ============================================================

def test_candidates(pipeline):
    """Test model with sample candidates"""
    
    print("\n" + "="*50)
    print("🧪 TESTING CANDIDATES")
    print("="*50)
    
    candidates = [
        {
            'name': 'Laraib Kaleem (You)',
            'resume': """MS Data Science Bahria University
                Pakistan published NLP researcher Springer
                Nature BERT T5 transformers PyTorch
                TensorFlow explainable AI XAI SHAP
                Samsung AI trainee machine learning
                deep learning python data science""",
            'job': 'Senior ML Engineer'
        },
        {
            'name': 'Fatima Khan (Bias Case)',
            'resume': """PhD Computer Science MIT
                8 years machine learning experience
                published 12 papers NLP conferences
                expert deep learning python tensorflow
                led teams amazon data science research""",
            'job': 'Senior ML Engineer'
        },
        {
            'name': 'Random Person (Unqualified)',
            'resume': """High school diploma
                worked grocery store cashier
                customer service retail sales
                no technical skills experience""",
            'job': 'Senior ML Engineer'
        },
        {
            'name': 'Ahmed Raza (Career Change)',
            'resume': """BS Information Technology
                self taught machine learning coursera
                3 years data analysis experience
                python SQL scikit-learn pandas
                built prediction models ecommerce""",
            'job': 'Data Scientist'
        }
    ]
    
    for candidate in candidates:
        result = predict_candidate(
            pipeline,
            candidate['resume'],
            candidate['job']
        )
        print(f"\nCandidate: {candidate['name']}")
        print(f"Job: {candidate['job']}")
        print(f"Decision: {result['decision']}")
        print(f"Probability: {result['hire_percentage']}%")
        print(f"Confidence: {result['confidence']}%")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    # Load data
    df = load_data()
    
    # Train model
    pipeline, X_train, X_test, y_test, metrics = (
        train_model(df)
    )
    
    # Setup SHAP
    try:
        explainer, features = setup_shap(pipeline, X_train)
    except Exception as e:
        print(f"⚠️ SHAP setup: {e}")
        print("Continuing without SHAP...")
    
    # Save model
    save_model(pipeline, metrics, X_train)
    
    # Test candidates
    test_candidates(pipeline)
    
    print("\n" + "="*50)
    print("✅ DAY 5 COMPLETE!")
    print("="*50)
    print("📌 Next: Phoenix Integration + Agent")

# """
# FairHire AI - BERT Training (CPU Optimized)
# Runs locally in VS Code without GPU
# """

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

# print("="*50)
# print("🚀 FAIRHIRE AI - LOCAL BERT TRAINING")
# print("="*50)
# print(f"✅ PyTorch: {torch.__version__}")
# print(f"✅ Device: CPU (local mode)")

# # ============================================================
# # CONFIG (CPU OPTIMIZED)
# # ============================================================

# CONFIG = {
#     'model_name': 'distilbert-base-uncased',
#     'max_length': 128,        # Reduced for CPU
#     'num_labels': 2,
#     'batch_size': 8,          # Reduced for CPU
#     'epochs': 2,              # Reduced for CPU
#     'learning_rate': 2e-5,
#     'warmup_steps': 50,
#     'weight_decay': 0.01,
#     'test_size': 0.2,
#     'random_seed': 42,
#     'sample_size': 500,       # Use 500 samples only
#     'model_save_path': 'models/bert_screener',
#     'metrics_save_path': 'models/model_metrics.json'
# }

# device = torch.device('cpu')
# torch.manual_seed(CONFIG['random_seed'])
# np.random.seed(CONFIG['random_seed'])

# print(f"✅ Config loaded")
# print(f"✅ Sample size: {CONFIG['sample_size']}")
# print(f"✅ Epochs: {CONFIG['epochs']}")
# print(f"✅ Batch size: {CONFIG['batch_size']}")

# # ============================================================
# # LOAD DATASET
# # ============================================================

# def load_dataset():
#     """Load training data"""
    
#     print(f"\n📂 Loading dataset...")
    
#     df = pd.read_csv(
#         'data/resumes_processed/training_data.csv'
#     )
    
#     # Sample for CPU speed
#     df_hired = df[df['hired']==1].sample(
#         n=CONFIG['sample_size']//2,
#         random_state=42
#     )
#     df_rejected = df[df['hired']==0].sample(
#         n=CONFIG['sample_size']//2,
#         random_state=42
#     )
#     df = pd.concat([df_hired, df_rejected]).sample(
#         frac=1, random_state=42
#     ).reset_index(drop=True)
    
#     print(f"✅ Samples loaded: {len(df)}")
#     print(f"✅ Hired: {df['hired'].sum()} "
#           f"({df['hired'].mean()*100:.1f}%)")
#     print(f"✅ Rejected: {(df['hired']==0).sum()} "
#           f"({(df['hired']==0).mean()*100:.1f}%)")
    
#     # Create input text
#     df['input_text'] = (
#         "RESUME: " +
#         df['resume_text'].fillna('').str[:200] +
#         " JOB: " +
#         df['job_description'].fillna('').str[:100]
#     )
    
#     return df

# # ============================================================
# # DATASET CLASS
# # ============================================================

# class ResumeDataset(Dataset):
    
#     def __init__(self, texts, labels, 
#                  tokenizer, max_length):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             str(self.texts[idx]),
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': (
#                 encoding['attention_mask'].flatten()
#             ),
#             'label': torch.tensor(
#                 self.labels[idx], dtype=torch.long
#             )
#         }

# # ============================================================
# # TRAIN ONE EPOCH
# # ============================================================

# def train_epoch(model, dataloader, optimizer, scheduler):
    
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     for batch_idx, batch in enumerate(dataloader):
        
#         optimizer.zero_grad()
        
#         outputs = model(
#             input_ids=batch['input_ids'],
#             attention_mask=batch['attention_mask'],
#             labels=batch['label']
#         )
        
#         loss = outputs.loss
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(), 1.0
#         )
        
#         optimizer.step()
#         scheduler.step()
        
#         total_loss += loss.item()
#         preds = torch.argmax(outputs.logits, dim=1)
#         correct += (preds == batch['label']).sum().item()
#         total += batch['label'].size(0)
        
#         # Show progress every 10 batches
#         if batch_idx % 10 == 0:
#             print(f"  Batch {batch_idx}/{len(dataloader)} "
#                   f"| Loss: {loss.item():.4f} "
#                   f"| Acc: {correct/total*100:.1f}%")
    
#     return total_loss/len(dataloader), correct/total

# # ============================================================
# # EVALUATE
# # ============================================================

# def evaluate(model, dataloader):
    
#     model.eval()
#     all_preds = []
#     all_labels = []
#     total_loss = 0
    
#     with torch.no_grad():
#         for batch in dataloader:
#             outputs = model(
#                 input_ids=batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#                 labels=batch['label']
#             )
#             total_loss += outputs.loss.item()
#             preds = torch.argmax(outputs.logits, dim=1)
#             all_preds.extend(preds.numpy())
#             all_labels.extend(batch['label'].numpy())
    
#     return {
#         'loss': total_loss/len(dataloader),
#         'accuracy': accuracy_score(all_labels, all_preds),
#         'precision': precision_score(
#             all_labels, all_preds, zero_division=0
#         ),
#         'recall': recall_score(
#             all_labels, all_preds, zero_division=0
#         ),
#         'f1': f1_score(
#             all_labels, all_preds, zero_division=0
#         ),
#         'predictions': all_preds,
#         'labels': all_labels
#     }

# # ============================================================
# # MAIN TRAINING
# # ============================================================

# def main():
    
#     # Load data
#     df = load_dataset()
    
#     # Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['input_text'].values,
#         df['hired'].values,
#         test_size=CONFIG['test_size'],
#         random_state=CONFIG['random_seed'],
#         stratify=df['hired'].values
#     )
#     print(f"\n✅ Train: {len(X_train)} samples")
#     print(f"✅ Test: {len(X_test)} samples")
    
#     # Load tokenizer
#     print(f"\n📥 Loading tokenizer...")
#     tokenizer = DistilBertTokenizer.from_pretrained(
#         CONFIG['model_name']
#     )
#     print("✅ Tokenizer loaded")
    
#     # Create datasets
#     train_ds = ResumeDataset(
#         X_train, y_train,
#         tokenizer, CONFIG['max_length']
#     )
#     test_ds = ResumeDataset(
#         X_test, y_test,
#         tokenizer, CONFIG['max_length']
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=CONFIG['batch_size'],
#         shuffle=True
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=CONFIG['batch_size'],
#         shuffle=False
#     )
    
#     print(f"✅ Train batches: {len(train_loader)}")
#     print(f"✅ Test batches: {len(test_loader)}")
    
#     # Load model
#     print(f"\n📥 Loading DistilBERT model...")
#     print("(First time downloads ~250MB, please wait...)")
#     model = DistilBertForSequenceClassification.from_pretrained(
#         CONFIG['model_name'],
#         num_labels=CONFIG['num_labels']
#     )
#     print("✅ Model loaded!")
    
#     # Setup optimizer & scheduler
#     optimizer = AdamW(
#         model.parameters(),
#         lr=CONFIG['learning_rate'],
#         weight_decay=CONFIG['weight_decay']
#     )
#     total_steps = len(train_loader) * CONFIG['epochs']
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=CONFIG['warmup_steps'],
#         num_training_steps=total_steps
#     )
    
#     # Create save directory
#     os.makedirs(CONFIG['model_save_path'], exist_ok=True)
    
#     # Training loop
#     print(f"\n🎯 Starting training...")
#     print(f"Epochs: {CONFIG['epochs']}")
#     print(f"Estimated time: 30-60 minutes on CPU")
#     print("-"*50)
    
#     best_accuracy = 0
#     history = []
    
#     for epoch in range(CONFIG['epochs']):
        
#         print(f"\n📈 EPOCH {epoch+1}/{CONFIG['epochs']}")
#         print("-"*30)
        
#         train_loss, train_acc = train_epoch(
#             model, train_loader, optimizer, scheduler
#         )
        
#         print(f"\n⏳ Evaluating...")
#         test_metrics = evaluate(model, test_loader)
        
#         print(f"\n📊 Epoch {epoch+1} Results:")
#         print(f"  Train Loss: {train_loss:.4f}")
#         print(f"  Train Acc:  {train_acc*100:.2f}%")
#         print(f"  Test Loss:  {test_metrics['loss']:.4f}")
#         print(f"  Test Acc:   "
#               f"{test_metrics['accuracy']*100:.2f}%")
#         print(f"  Precision:  "
#               f"{test_metrics['precision']*100:.2f}%")
#         print(f"  Recall:     "
#               f"{test_metrics['recall']*100:.2f}%")
#         print(f"  F1:         "
#               f"{test_metrics['f1']*100:.2f}%")
        
#         history.append({
#             'epoch': epoch+1,
#             'train_loss': round(train_loss, 4),
#             'train_accuracy': round(train_acc, 4),
#             'test_accuracy': round(
#                 test_metrics['accuracy'], 4
#             ),
#             'test_f1': round(test_metrics['f1'], 4)
#         })
        
#         # Save best model
#         if test_metrics['accuracy'] > best_accuracy:
#             best_accuracy = test_metrics['accuracy']
#             print(f"\n💾 New best model! Saving...")
#             model.save_pretrained(CONFIG['model_save_path'])
#             tokenizer.save_pretrained(
#                 CONFIG['model_save_path']
#             )
#             print(f"✅ Model saved to "
#                   f"{CONFIG['model_save_path']}/")
    
#     # Final report
#     print("\n" + "="*50)
#     print("📊 FINAL RESULTS")
#     print("="*50)
    
#     final = evaluate(model, test_loader)
    
#     print(f"\nAccuracy:  {final['accuracy']*100:.2f}%")
#     print(f"Precision: {final['precision']*100:.2f}%")
#     print(f"Recall:    {final['recall']*100:.2f}%")
#     print(f"F1 Score:  {final['f1']*100:.2f}%")
    
#     print("\nClassification Report:")
#     print(classification_report(
#         final['labels'],
#         final['predictions'],
#         target_names=['Rejected', 'Hired']
#     ))
    
#     # Save metrics
#     metrics = {
#         'config': CONFIG,
#         'history': history,
#         'final_metrics': {
#             'accuracy': round(final['accuracy'], 4),
#             'precision': round(final['precision'], 4),
#             'recall': round(final['recall'], 4),
#             'f1': round(final['f1'], 4)
#         },
#         'best_accuracy': round(best_accuracy, 4),
#         'train_samples': len(X_train),
#         'test_samples': len(X_test),
#         'device': 'cpu'
#     }
    
#     with open(CONFIG['metrics_save_path'], 'w') as f:
#         json.dump(metrics, f, indent=2)
    
#     print(f"\n✅ Metrics saved!")
#     print(f"✅ Model saved!")
#     print(f"\n🏆 Best Accuracy: {best_accuracy*100:.2f}%")
#     print("\n✅ TRAINING COMPLETE!")
#     print("📌 Next: Phoenix Integration + Agent")
    
#     return model, tokenizer, metrics

# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":
#     main()