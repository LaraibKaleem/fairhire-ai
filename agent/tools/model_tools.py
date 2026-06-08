"""
FairHire AI - Model Analysis Tools
Tools the agent uses to analyze hiring decisions
"""

import pickle
import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load trained Random Forest model"""
    model_path = "models/bert_screener/model.pkl"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}"
        )
    
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    return pipeline

# Global model instance
try:
    MODEL = load_model()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model load warning: {e}")
    MODEL = None

# ============================================================
# TOOL 1: SCREEN RESUME
# ============================================================

def screen_resume(resume_text: str, job_title: str) -> dict:
    """
    Screen a resume against a job position
    Returns hiring decision and probability
    """
    
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    input_text = f"{resume_text} {job_title}"
    
    prob = MODEL.predict_proba([input_text])[0][1]
    prediction = int(prob > 0.5)
    
    return {
        "decision": "HIRED" if prediction == 1 else "REJECTED",
        "hire_probability": round(prob, 3),
        "hire_percentage": round(prob * 100, 1),
        "confidence": round(max(prob, 1-prob) * 100, 1),
        "job_title": job_title
    }

# ============================================================
# TOOL 2: EXPLAIN DECISION (SHAP)
# ============================================================

def explain_decision(
    resume_text: str, 
    job_title: str,
    top_n: int = 10
) -> dict:
    """
    Explain why a candidate was hired or rejected
    Uses SHAP to identify key features
    """
    
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    tfidf = MODEL.named_steps['tfidf']
    rf = MODEL.named_steps['classifier']
    
    input_text = f"{resume_text} {job_title}"
    
    # Get TF-IDF features
    X = tfidf.transform([input_text])
    feature_names = tfidf.get_feature_names_out()
    n_features = len(feature_names)
    
    # Get SHAP values
    try:
        explainer = shap.TreeExplainer(rf)
        X_array = X.toarray()
        shap_values = explainer.shap_values(X_array)
        
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1][0]).flatten()
        else:
            sv = np.array(shap_values[0]).flatten()
        
        sv = sv[:n_features]
        
        # Get top positive (hire) features
        pos_idx = np.argsort(sv)[-top_n:][::-1]
        positive_features = [
            {
                "feature": str(feature_names[i]),
                "impact": round(float(sv[i]), 4),
                "direction": "supports_hire"
            }
            for i in pos_idx if sv[i] > 0
        ]
        
        # Get top negative (reject) features
        neg_idx = np.argsort(sv)[:top_n]
        negative_features = [
            {
                "feature": str(feature_names[i]),
                "impact": round(float(sv[i]), 4),
                "direction": "against_hire"
            }
            for i in neg_idx if sv[i] < 0
        ]

        # Get ALL features sorted by absolute impact
        all_features = []
        nonzero_idx = np.nonzero(sv)[0]
        for idx in nonzero_idx:
            all_features.append({
                "feature": str(feature_names[idx]),
                "impact": round(float(sv[idx]), 4)
            })

        all_features.sort(key=lambda x: x['impact'], reverse=True)

        positive = [f for f in all_features if f['impact'] > 0][:5]
        negative = [f for f in all_features if f['impact'] < 0][-5:]
        negative.reverse()

        return {
            "positive_features": positive,
            "negative_features": negative,
            "top_positive": positive[0]['feature'] if positive else "none",
            "top_negative": negative[0]['feature'] if negative else "none"
        
        # return {
        #     "positive_features": positive_features[:5],
        #     "negative_features": negative_features[:5],
        #     "explanation": generate_explanation(
        #         positive_features[:3],
        #         negative_features[:3]
        #     )
        }
    
    except Exception as e:
        return {"error": f"SHAP error: {str(e)}"}

def generate_explanation(pos_features, neg_features):
    """Generate human-readable explanation"""
    
    explanation = []
    
    if pos_features:
        pos_words = [f['feature'] for f in pos_features]
        explanation.append(
            f"Positive factors: {', '.join(pos_words)}"
        )
    
    if neg_features:
        neg_words = [f['feature'] for f in neg_features]
        explanation.append(
            f"Negative factors: {', '.join(neg_words)}"
        )
    
    return ". ".join(explanation)

# ============================================================
# TOOL 3: DETECT BIAS
# ============================================================

def detect_bias(
    resume_text: str,
    decision: str,
    probability: float
) -> dict:
    """
    Detect potential bias in hiring decision
    Checks for gender, nationality, education, career gap bias
    """
    
    resume_lower = resume_text.lower()
    bias_flags = []
    bias_details = []
    
    # Gender bias
    female_names = [
        "fatima", "sarah", "emily", "priya",
        "maria", "laraib", "aisha", "maryam",
        "zara", "sana", "hiba", "noor", "ayesha"
    ]
    if any(n in resume_lower for n in female_names):
        if decision == "REJECTED" and probability < 0.6:
            bias_flags.append("gender_bias")
            bias_details.append(
                "Female candidate rejected with low confidence"
            )
    
    # Career gap bias
    gap_keywords = [
        "career break", "maternity", "paternity",
        "family care", "sabbatical", "gap year",
        "time off", "personal leave", "health recovery"
    ]
    if any(g in resume_lower for g in gap_keywords):
        if decision == "REJECTED":
            bias_flags.append("career_gap_bias")
            bias_details.append(
                "Candidate with career gap rejected"
            )
    
    # Education bias
    non_prestige = [
        "community college", "local institute",
        "online university", "bahria", "state college"
    ]
    prestige = [
        "mit", "stanford", "harvard",
        "berkeley", "cmu", "oxford"
    ]
    if (any(s in resume_lower for s in non_prestige) and
        not any(s in resume_lower for s in prestige)):
        if decision == "REJECTED":
            bias_flags.append("education_bias")
            bias_details.append(
                "Non-prestige university candidate rejected"
            )
    
    # Nationality bias
    countries = [
        "pakistan", "india", "nigeria",
        "bangladesh", "vietnam", "egypt"
    ]
    if any(c in resume_lower for c in countries):
        if decision == "REJECTED" and probability < 0.55:
            bias_flags.append("nationality_bias")
            bias_details.append(
                "International candidate rejected with low confidence"
            )
    
    # Title mismatch bias
    if "no direct" in resume_lower:
        if decision == "REJECTED":
            bias_flags.append("title_mismatch_bias")
            bias_details.append(
                "Candidate rejected possibly due to title mismatch"
            )
    
    # Generate recommendation
    if bias_flags:
        recommendation = (
            f"⚠️ BIAS DETECTED: {', '.join(bias_flags)}. "
            f"Recommend manual review. "
            f"Details: {'. '.join(bias_details)}"
        )
        final_decision = "FLAGGED FOR REVIEW"
    else:
        recommendation = "✅ No bias detected. Decision appears fair."
        final_decision = decision
    
    return {
        "bias_detected": len(bias_flags) > 0,
        "bias_flags": bias_flags,
        "bias_details": bias_details,
        "recommendation": recommendation,
        "final_decision": final_decision
    }

# ============================================================
# TOOL 4: GET MODEL METRICS
# ============================================================

def get_model_metrics() -> dict:
    """
    Get current model performance metrics
    Returns accuracy, precision, recall, F1
    """
    
    metrics_path = "models/model_metrics.json"
    
    if not Path(metrics_path).exists():
        return {"error": "Metrics file not found"}
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    final = metrics.get('final_metrics', {})
    
    return {
        "accuracy": round(
            final.get('accuracy', 0) * 100, 2
        ),
        "precision": round(
            final.get('precision', 0) * 100, 2
        ),
        "recall": round(
            final.get('recall', 0) * 100, 2
        ),
        "f1_score": round(
            final.get('f1', 0) * 100, 2
        ),
        "model_type": metrics.get(
            'model_type', 'Random Forest'
        ),
        "training_samples": metrics.get(
            'training_samples', 0
        ),
        "status": "healthy" if final.get(
            'accuracy', 0
        ) > 0.75 else "degraded"
    }

# ============================================================
# TOOL 5: ANALYZE BATCH
# ============================================================

def analyze_batch(candidates: list) -> dict:
    """
    Analyze multiple candidates for bias patterns
    Returns aggregate bias statistics
    """
    
    results = []
    bias_count = 0
    hired_count = 0
    
    for candidate in candidates:
        # Screen
        screen = screen_resume(
            candidate['resume'],
            candidate['job']
        )
        
        # Detect bias
        bias = detect_bias(
            candidate['resume'],
            screen['decision'],
            screen['hire_probability']
        )
        
        if bias['bias_detected']:
            bias_count += 1
        if screen['decision'] == 'HIRED':
            hired_count += 1
        
        results.append({
            'name': candidate.get('name', 'Unknown'),
            'decision': screen['decision'],
            'probability': screen['hire_percentage'],
            'bias_detected': bias['bias_detected'],
            'bias_flags': bias['bias_flags']
        })
    
    total = len(candidates)
    
    return {
        "total_candidates": total,
        "hired": hired_count,
        "rejected": total - hired_count,
        "hire_rate": round(hired_count/total*100, 1),
        "bias_detected_count": bias_count,
        "bias_rate": round(bias_count/total*100, 1),
        "results": results,
        "summary": (
            f"Analyzed {total} candidates. "
            f"Hired: {hired_count} ({hired_count/total*100:.1f}%). "
            f"Bias detected in {bias_count} cases "
            f"({bias_count/total*100:.1f}%)."
        )
    }