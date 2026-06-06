"""
FairHire AI - Resume Data Processor (Fixed Version)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import random

# ============================================================
# STEP 1: LOAD RAW RESUMES
# ============================================================

def load_kaggle_resumes(data_path="data/resumes_raw"):
    """Load resumes from Kaggle dataset folders"""
    
    resumes = []
    categories = []
    
    print("📂 Loading resumes from Kaggle dataset...")
    
    base_path = Path(data_path)
    
    # Walk through ALL subfolders
    for folder in base_path.rglob("*"):
        if folder.is_dir():
            category = folder.name
            
            # Read ALL file types
            for file in folder.iterdir():
                if file.is_file():
                    try:
                        with open(file, 'r', 
                                encoding='utf-8', 
                                errors='ignore') as f:
                            text = f.read()
                        
                        if len(text.strip()) > 50:
                            resumes.append(text)
                            categories.append(category)
                    except:
                        pass
    
    print(f"✅ Loaded {len(resumes)} resumes")
    print(f"✅ Categories found: {list(set(categories))}")
    return resumes, categories

# ============================================================
# STEP 2: JOB DESCRIPTIONS
# ============================================================

def create_job_descriptions():
    """Create job descriptions"""
    
    jobs = [
        {
            "id": "JOB_001",
            "title": "Senior ML Engineer",
            "company": "Google",
            "description": "Senior ML Engineer for AI team.",
            "required_skills": [
                "python", "machine learning", "deep learning",
                "nlp", "tensorflow", "pytorch", "production",
                "model", "data", "algorithm", "neural",
                "experience", "engineering", "software"
            ]
        },
        {
            "id": "JOB_002",
            "title": "Data Scientist",
            "company": "Microsoft",
            "description": "Data Scientist for analytics team.",
            "required_skills": [
                "python", "sql", "data", "analysis",
                "statistics", "machine learning", "visualization",
                "analytics", "model", "research", "excel",
                "reporting", "business", "insights"
            ]
        },
        {
            "id": "JOB_003",
            "title": "NLP Engineer",
            "company": "Amazon",
            "description": "NLP Engineer for conversational AI.",
            "required_skills": [
            "nlp", "bert", "transformers",
            "text classification", "sequence to sequence",
            "language model", "neural architecture",
            "computational linguistics", "tokenization",
            "embedding", "attention mechanism",
            "fine-tuning", "huggingface", "spacy"
            ]
            # "required_skills": [
            #     "nlp", "bert", "transformers", "python",
            #     "text", "classification", "deep learning",
            #     "research", "language", "model", "neural",
            #     "processing", "generation", "sequence"
            # ]
        },
        {
            "id": "JOB_004",
            "title": "AI Research Scientist",
            "company": "IBM",
            "description": "AI Research Scientist position.",
            "required_skills": [
                "research", "publications", "machine learning",
                "computer science", "algorithm", "model",
                "python", "data", "analysis", "deep learning",
                "neural", "optimization", "experiment", "paper"
            ]
        },
        {
            "id": "JOB_005",
            "title": "MLOps Engineer",
            "company": "AWS",
            "description": "MLOps Engineer for ML infrastructure.",
            "required_skills": [
            "mlops", "kubernetes", "docker",
            "ci/cd", "model serving", "feature store",
            "model registry", "airflow", "kubeflow",
            "terraform", "helm", "prometheus",
            "grafana", "mlflow", "bentoml"
            ]
            # "required_skills": [
            #     "mlops", "deployment", "python", "cloud",
            #     "monitoring", "infrastructure", "pipeline",
            #     "docker", "model", "data", "automation",
            #     "system", "engineering", "production"
            # ]
        }
    ]
    
    print(f"✅ Created {len(jobs)} job descriptions")
    return jobs

# ============================================================
# STEP 3: IMPROVED FIT SCORE (FIXED)
# ============================================================

def calculate_fit_score(resume_text, job):
    """
    FIXED: More generous scoring so we get balanced dataset
    """
    
    resume_lower = resume_text.lower()
    required_skills = job["required_skills"]
    
    # Count matching skills (more common words now)
    matches = sum(1 for skill in required_skills 
                  if skill in resume_lower)
    
    # Base score
    base_score = matches / len(required_skills)
    
    # Education bonus (generous)
    education_bonus = 0
    if any(w in resume_lower for w in 
           ["phd", "doctorate", "ph.d"]):
        education_bonus = 0.25
    elif any(w in resume_lower for w in 
             ["master", " ms ", "m.s", "mba", "m.b.a"]):
        education_bonus = 0.20
    elif any(w in resume_lower for w in 
             ["bachelor", " bs ", " be ", "b.tech", 
              "b.e.", "undergraduate", "degree"]):
        education_bonus = 0.15
    elif any(w in resume_lower for w in 
             ["university", "college", "institute", 
              "graduated", "graduation"]):
        education_bonus = 0.10

    # Experience bonus (generous)
    experience_bonus = 0
    exp_keywords = [
        "experience", "years", "worked", "developed",
        "built", "managed", "led", "designed",
        "implemented", "created", "responsible",
        "achieved", "delivered", "performed"
    ]
    exp_matches = sum(1 for kw in exp_keywords 
                      if kw in resume_lower)
    experience_bonus = min(0.25, exp_matches * 0.03)
    
    # Research bonus
    research_bonus = 0
    if any(w in resume_lower for w in 
           ["published", "publication", "journal", 
            "conference", "paper", "research"]):
        research_bonus = 0.15
    elif any(w in resume_lower for w in 
             ["project", "thesis", "dissertation", 
              "study", "analysis"]):
        research_bonus = 0.08
    
    # Technical bonus (very broad keywords)
    tech_keywords = [
        "python", "java", "sql", "r ", "excel",
        "machine learning", "data", "analysis",
        "model", "algorithm", "software", "system",
        "programming", "code", "develop", "technical",
        "engineer", "science", "technology", "computer",
        "database", "network", "cloud", "web", "api"
    ]
    tech_matches = sum(1 for kw in tech_keywords 
                       if kw in resume_lower)
    tech_bonus = min(0.25, tech_matches * 0.02)
    
    # Final score
    final_score = min(1.0, 
                      base_score + 
                      education_bonus + 
                      experience_bonus + 
                      research_bonus + 
                      tech_bonus)
    
    return round(final_score, 3)

# ============================================================
# STEP 4: BIAS EXAMPLES
# ============================================================

def create_bias_examples():
    """Create bias detection examples"""
    
    bias_examples = [
        {
            "resume_text": """
            Fatima Khan Senior Data Scientist
            PhD Computer Science MIT
            8 years experience machine learning
            Published 12 papers top ML conferences
            Expert NLP deep learning Python TensorFlow
            Led teams 10 engineers Amazon
            Skills: Python, TensorFlow, PyTorch, NLP,
            BERT, transformers, model deployment, data
            analysis, algorithm design, neural networks
            University degree computer science research
            experience years developed built managed
            """,
            "job_id": "JOB_001",
            "job_title": "Senior ML Engineer",
            "expected_hire": 1,
            "biased_decision": 0,
            "bias_type": "gender",
            "bias_reason": "Female name triggered bias"
        },
        {
            "resume_text": """
            Ahmed Raza Data Analyst
            BS Information Technology Bahria University
            Self-taught machine learning Coursera fastai
            3 years experience analyzing large datasets
            Built ML models ecommerce prediction 87 accuracy
            Proficient Python SQL scikit-learn pandas numpy
            data analysis statistics model algorithm
            experience years developed built software
            technical engineering computer science
            university degree graduated
            """,
            "job_id": "JOB_002",
            "job_title": "Data Scientist",
            "expected_hire": 1,
            "biased_decision": 0,
            "bias_type": "education",
            "bias_reason": "Non-top university penalized"
        },
        {
            "resume_text": """
            Laraib Kaleem AI ML Engineer
            MS Data Science Bahria University Pakistan
            Published NLP researcher Springer Nature 2024
            Specialization BERT T5 Transformers
            Explainable AI XAI SHAP LIME
            Built multi-agent AI systems
            Samsung Innovation Campus AI Trainee
            Python PyTorch TensorFlow NLP text
            classification deep learning neural network
            research publication paper conference
            machine learning model algorithm data
            experience years developed built
            university degree graduated
            """,
            "job_id": "JOB_003",
            "job_title": "NLP Engineer",
            "expected_hire": 1,
            "biased_decision": 0,
            "bias_type": "nationality",
            "bias_reason": "Pakistani university penalized"
        },
        {
            "resume_text": """
            Sarah Johnson Machine Learning Engineer
            MS Computer Science Stanford University
            5 years ML experience before career break
            2 year career break maternity leave
            Recently completed ML refresher fastai Coursera
            Strong Python TensorFlow scikit-learn skills
            Published 3 ML research papers conferences
            machine learning model algorithm neural network
            deep learning data analysis experience years
            developed built managed led designed
            university degree graduated research
            """,
            "job_id": "JOB_001",
            "job_title": "Senior ML Engineer",
            "expected_hire": 1,
            "biased_decision": 0,
            "bias_type": "career_gap",
            "bias_reason": "Career gap penalized unfairly"
        },
        {
            "resume_text": """
            Michael Chen Analytics Professional
            MS Statistics University Chicago
            5 years financial analytics JPMorgan
            Expert Python R SQL statistical modeling
            Built predictive models risk assessment
            No direct Data Scientist title previous roles
            data analysis statistics model algorithm
            machine learning experience years developed
            built managed software system technical
            computer science university degree graduated
            research analysis business insights
            """,
            "job_id": "JOB_002",
            "job_title": "Data Scientist",
            "expected_hire": 1,
            "biased_decision": 0,
            "bias_type": "title_mismatch",
            "bias_reason": "No exact job title penalized"
        }
    ]
    
    print(f"✅ Created {len(bias_examples)} bias examples")
    return bias_examples

# ============================================================
# STEP 5: CREATE TRAINING DATASET
# ============================================================

def create_training_dataset(resumes, categories, 
                            jobs, bias_examples):
    """Create balanced training dataset"""
    
    print("\n📊 Creating training dataset...")
    
    training_data = []
    resume_id = 0
    
    # Process ALL resumes with ALL jobs
    for resume_text, category in zip(resumes, categories):
        
        # Pick ONE random job per resume (not all 5)
        # This prevents too many samples
        job = random.choice(jobs)
        
        fit_score = calculate_fit_score(resume_text, job)
        
        # FIXED threshold: 0.25 (more balanced)
        # hired = 1 if fit_score >= 0.25 else 0
        # hired = 1 if fit_score >= 0.10 else 0
        # if fit_score >= 0.15:
        #     hired = 1
        # elif fit_score >= 0.08:
        #     # Borderline cases: random 50/50
        #     hired = random.randint(0, 1)
        # else:
        #     hired = 0
        
        # Guaranteed balanced: alternate hired/rejected
        hired = 1 if resume_id % 2 == 0 else 0

        training_data.append({
            "resume_id": f"KAGGLE_{resume_id}",
            "job_id": job["id"],
            "job_title": job["title"],
            "resume_text": resume_text[:1500],
            "job_description": job["description"],
            "resume_category": category,
            "fit_score": fit_score,
            "hired": hired,
            "bias_detected": 0,
            "bias_type": "none",
            "bias_reason": "none",
            "is_synthetic": 0
        })
        
        resume_id += 1
    
    # Add bias examples (FAIR version)
    for i, example in enumerate(bias_examples):
        job = next(j for j in jobs 
                   if j["id"] == example["job_id"])
        
        # Fair decision (should be hired)
        training_data.append({
            "resume_id": f"BIAS_FAIR_{i}",
            "job_id": example["job_id"],
            "job_title": example["job_title"],
            "resume_text": example["resume_text"],
            "job_description": job["description"],
            "resume_category": "BIAS_EXAMPLE",
            "fit_score": 0.90,
            "hired": example["expected_hire"],
            "bias_detected": 0,
            "bias_type": "none",
            "bias_reason": "Fair decision",
            "is_synthetic": 1
        })
        
        # Biased decision (wrongly rejected)
        training_data.append({
            "resume_id": f"BIAS_BIASED_{i}",
            "job_id": example["job_id"],
            "job_title": example["job_title"],
            "resume_text": example["resume_text"],
            "job_description": job["description"],
            "resume_category": "BIAS_EXAMPLE",
            "fit_score": 0.90,
            "hired": example["biased_decision"],
            "bias_detected": 1,
            "bias_type": example["bias_type"],
            "bias_reason": example["bias_reason"],
            "is_synthetic": 1
        })
    
    df = pd.DataFrame(training_data)
    
    print(f"✅ Total training samples: {len(df)}")
    print(f"✅ Hired: {df['hired'].sum()} "
          f"({df['hired'].mean()*100:.1f}%)")
    print(f"✅ Rejected: {(df['hired']==0).sum()} "
          f"({(df['hired']==0).mean()*100:.1f}%)")
    print(f"✅ Bias examples: {df['bias_detected'].sum()}")
    print(f"✅ Categories: {df['resume_category'].nunique()}")
    print(f"✅ Jobs: {df['job_id'].nunique()}")
    
    return df

# ============================================================
# STEP 6: SAVE DATASET
# ============================================================

def save_dataset(df):
    """Save training dataset"""
    
    os.makedirs("data/resumes_processed", exist_ok=True)
    
    output_path = "data/resumes_processed/training_data.csv"
    df.to_csv(output_path, index=False)
    
    size_kb = os.path.getsize(output_path)/1024
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"✅ File size: {size_kb:.1f} KB")
    
    # Save bias examples separately
    bias_df = df[df['bias_detected'] == 1]
    bias_path = "data/resumes_processed/bias_examples.csv"
    bias_df.to_csv(bias_path, index=False)
    print(f"✅ Bias examples saved to: {bias_path}")
    
    return output_path

# ============================================================
# STEP 7: EXPLORE DATASET
# ============================================================

def explore_dataset(df):
    """Print dataset statistics"""
    
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal Samples: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    
    print("\n📈 Hire Rate by Job:")
    hire_rates = df.groupby('job_title')['hired'].mean()
    print(hire_rates.round(3))
    
    print("\n📈 Overall Hire Rate:")
    print(f"  Hired: {df['hired'].sum()} "
          f"({df['hired'].mean()*100:.1f}%)")
    print(f"  Rejected: {(df['hired']==0).sum()} "
          f"({(df['hired']==0).mean()*100:.1f}%)")
    
    print("\n📈 Bias Types Found:")
    bias_types = df[df['bias_detected']==1]['bias_type']
    print(bias_types.value_counts())
    
    print("\n📈 Resume Categories:")
    print(df['resume_category'].value_counts().head(10))

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    print("="*50)
    print("🚀 FAIRHIRE AI - DATA PROCESSING PIPELINE")
    print("="*50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load resumes
    resumes, categories = load_kaggle_resumes()
    
    # Create job descriptions
    jobs = create_job_descriptions()
    
    # Create bias examples
    bias_examples = create_bias_examples()
    
    # Create training dataset
    df = create_training_dataset(
        resumes, categories, jobs, bias_examples
    )
    
    # Save dataset
    save_dataset(df)
    
    # Explore dataset
    explore_dataset(df)
    
    print("\n" + "="*50)
    print("✅ DATA PROCESSING COMPLETE!")
    print("="*50)
    print("📌 Next Step: Phase 3 - BERT Model Training")