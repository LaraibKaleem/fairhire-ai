"""
FairHire AI - Resume Data Processor (BALANCED & UNBIASED VERSION)
Fixed: Class balance, uniform thresholds, job-resume matching, more bias examples
CORRECTED: Balance enforcement now works by ranking scores instead of threshold
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
# STEP 2: JOB DESCRIPTIONS (EXPANDED WITH NON-TECH)
# ============================================================

def create_job_descriptions():
    """Create job descriptions - NOW INCLUDING NON-TECH ROLES"""

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
            ],
            "match_categories": [
                'INFORMATION-TECHNOLOGY', 'ENGINEERING', 
                'DATA-SCIENCE', 'RESEARCH'
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
            ],
            "match_categories": [
                'INFORMATION-TECHNOLOGY', 'ENGINEERING', 
                'DATA-SCIENCE', 'BANKING', 'FINANCE',
                'CONSULTANT', 'BUSINESS-DEVELOPMENT'
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
            ],
            "match_categories": [
                'INFORMATION-TECHNOLOGY', 'ENGINEERING', 
                'DATA-SCIENCE', 'DIGITAL-MEDIA'
            ]
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
            ],
            "match_categories": [
                'INFORMATION-TECHNOLOGY', 'ENGINEERING', 
                'DATA-SCIENCE', 'CONSULTANT'
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
            ],
            "match_categories": [
                'INFORMATION-TECHNOLOGY', 'ENGINEERING', 
                'DATA-SCIENCE', 'BANKING', 'FINANCE'
            ]
        },
        # NON-TECH JOBS - prevents tech-only bias
        {
            "id": "JOB_006",
            "title": "Executive Chef",
            "company": "Marriott",
            "description": "Lead kitchen operations and menu design.",
            "required_skills": [
                "cooking", "culinary", "kitchen", "menu",
                "food safety", "staff management", "inventory",
                "cuisine", "restaurant", "hospitality",
                "leadership", "planning", "quality"
            ],
            "match_categories": ['CHEF', 'HOSPITALITY', 'FOOD']
        },
        {
            "id": "JOB_007",
            "title": "UX Designer",
            "company": "Adobe",
            "description": "Design user experiences for products.",
            "required_skills": [
                "design", "ui", "ux", "figma", "sketch",
                "prototyping", "user research", "wireframing",
                "adobe", "creative", "visual", "branding"
            ],
            "match_categories": ['DESIGNER', 'ARTS', 'DIGITAL-MEDIA']
        },
        {
            "id": "JOB_008",
            "title": "Fitness Trainer",
            "company": "Equinox",
            "description": "Personal training and fitness programs.",
            "required_skills": [
                "fitness", "training", "nutrition", "exercise",
                "health", "wellness", "coaching", "sports",
                "motivation", "planning", "assessment"
            ],
            "match_categories": ['FITNESS', 'HEALTHCARE', 'SPORTS']
        }
    ]

    print(f"✅ Created {len(jobs)} job descriptions (tech + non-tech)")
    return jobs

# ============================================================
# STEP 3: UNBIASED FIT SCORE (UNIFORM THRESHOLD)
# ============================================================

def calculate_fit_score(resume_text, job):
    """
    BALANCED: Uniform scoring for ALL candidates
    No category-based discrimination in thresholds
    """

    resume_lower = resume_text.lower()
    required_skills = job["required_skills"]

    # Count matching skills
    matches = sum(1 for skill in required_skills 
                  if skill in resume_lower)

    # Base score (normalized)
    base_score = matches / len(required_skills)

    # Education bonus (same for ALL)
    education_bonus = 0
    if any(w in resume_lower for w in 
           ["phd", "doctorate", "ph.d"]):
        education_bonus = 0.15
    elif any(w in resume_lower for w in 
             ["master", " ms ", "m.s", "mba", "m.b.a"]):
        education_bonus = 0.12
    elif any(w in resume_lower for w in 
             ["bachelor", " bs ", " be ", "b.tech", 
              "b.e.", "undergraduate", "degree"]):
        education_bonus = 0.10
    elif any(w in resume_lower for w in 
             ["university", "college", "institute", 
              "graduated", "graduation", "diploma"]):
        education_bonus = 0.08

    # Experience bonus (same for ALL)
    experience_bonus = 0
    exp_keywords = [
        "experience", "years", "worked", "developed",
        "built", "managed", "led", "designed",
        "implemented", "created", "responsible",
        "achieved", "delivered", "performed"
    ]
    exp_matches = sum(1 for kw in exp_keywords 
                      if kw in resume_lower)
    experience_bonus = min(0.20, exp_matches * 0.025)

    # Research bonus
    research_bonus = 0
    if any(w in resume_lower for w in 
           ["published", "publication", "journal", 
            "conference", "paper", "research", "patent"]):
        research_bonus = 0.12
    elif any(w in resume_lower for w in 
             ["project", "thesis", "dissertation", 
              "study", "analysis", "investigation"]):
        research_bonus = 0.06

    # Technical bonus (broad, applies to ALL fields)
    tech_bonus = 0
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
    tech_bonus = min(0.15, tech_matches * 0.015)

    # Final score (capped at 1.0)
    final_score = min(1.0, 
                      base_score + 
                      education_bonus + 
                      experience_bonus + 
                      research_bonus + 
                      tech_bonus)

    return round(final_score, 3)

# ============================================================
# STEP 4: EXPANDED BIAS EXAMPLES (20% of dataset)
# ============================================================

def create_bias_examples():
    """Create comprehensive bias detection examples"""

    bias_templates = [
        {
            "bias_type": "gender",
            "bias_reason": "Female name triggered gender bias",
            "details": ["Fatima Khan", "Sarah Johnson", "Emily Chen", "Priya Sharma", "Maria Garcia"]
        },
        {
            "bias_type": "education",
            "bias_reason": "Non-top university penalized unfairly",
            "details": ["Bahria University", "State College", "Community College", "Online University", "Local Institute"]
        },
        {
            "bias_type": "nationality",
            "bias_reason": "International background penalized",
            "details": ["Pakistan", "India", "Nigeria", "Brazil", "Vietnam"]
        },
        {
            "bias_type": "career_gap",
            "bias_reason": "Career gap penalized unfairly",
            "details": ["maternity leave", "family care", "health recovery", "sabbatical", "relocation"]
        },
        {
            "bias_type": "title_mismatch",
            "bias_reason": "No exact job title in previous roles",
            "details": ["Analyst", "Specialist", "Coordinator", "Associate", "Generalist"]
        },
        {
            "bias_type": "age",
            "bias_reason": "Age discrimination detected",
            "details": ["20+ years experience", "recent graduate", "career changer", "senior professional", "young professional"]
        },
        {
            "bias_type": "race",
            "bias_reason": "Racial bias in evaluation",
            "details": ["ethnic name", "minority background", "immigrant status", "non-native speaker", "diverse candidate"]
        }
    ]

    bias_examples = []

    # Generate 80 pairs (fair + biased) = 160 examples
    for i in range(80):
        template = random.choice(bias_templates)
        job_id = random.choice(["JOB_001", "JOB_002", "JOB_003", "JOB_004", "JOB_005"])

        # All templates use "details" key - no KeyError possible
        detail = random.choice(template["details"])

        resume_text = f"""
        Candidate Name - Senior Professional
        PhD Computer Science - Relevant University
        8 years experience in relevant field
        Published multiple papers in top conferences
        Expert in required skills and technologies
        Led teams of engineers at major companies
        Strong background in: Python, ML, Data Analysis
        {detail} mentioned in background
        University degree, graduated with honors
        Experience years developed built managed led
        """

        bias_examples.append({
            "resume_text": resume_text,
            "job_id": job_id,
            "expected_hire": 1,
            "biased_decision": 0,
            "bias_type": template["bias_type"],
            "bias_reason": template["bias_reason"]
        })

    print(f"✅ Created {len(bias_examples)} bias examples (80 fair + 80 biased = 160 rows)")
    return bias_examples

# ============================================================
# STEP 5: BALANCED TRAINING DATASET - CORRECTED
# ============================================================

def create_training_dataset(resumes, categories, jobs, bias_examples):
    """Create balanced training dataset with WORKING balance enforcement"""

    print("\n📊 Creating training dataset...")

    # Step 1: Calculate ALL scores first (without labels)
    scored_resumes = []
    for resume_text, category in zip(resumes, categories):
        matching_jobs = [j for j in jobs if category in j.get("match_categories", [])]
        if not matching_jobs:
            matching_jobs = jobs
        job = random.choice(matching_jobs)

        fit_score = calculate_fit_score(resume_text, job)
        scored_resumes.append({
            'resume_text': resume_text,
            'category': category,
            'job': job,
            'fit_score': fit_score
        })

    # Step 2: Sort by fit_score (descending)
    scored_resumes.sort(key=lambda x: x['fit_score'], reverse=True)

    # Step 3: Force 50/50 balance by selecting top 50% as hired
    total = len(scored_resumes)
    hired_count = total // 2  # Exactly 50%

    print(f"   Total resumes: {total}")
    print(f"   Will hire: {hired_count} (top 50% by score)")
    print(f"   Will reject: {total - hired_count} (bottom 50% by score)")

    # Step 4: Assign labels based on ranking
    training_data = []
    for i, item in enumerate(scored_resumes):
        hired = 1 if i < hired_count else 0

        training_data.append({
            "resume_id": f"KAGGLE_{i}",
            "job_id": item['job']["id"],
            "job_title": item['job']["title"],
            "resume_text": item['resume_text'][:1500],
            "job_description": item['job']["description"],
            "resume_category": item['category'],
            "fit_score": item['fit_score'],
            "hired": hired,
            "bias_detected": 0,
            "bias_type": "none",
            "bias_reason": "none",
            "is_synthetic": 0
        })

    # Step 5: Add bias examples (80 fair + 80 biased = 160)
    for i, example in enumerate(bias_examples):
        job = next(j for j in jobs if j["id"] == example["job_id"])
        # Look up title and description from the job dict
        job_title = job["title"]
        job_description = job["description"]

        # "job_title": example["job_title"],#"job_description": job["description"]
        # Fair decision
        training_data.append({
            "resume_id": f"BIAS_FAIR_{i}",
            "job_id": example["job_id"],
            "job_title":job_title,
            "resume_text": example["resume_text"],
            "job_description": job_description,
            "resume_category": "BIAS_EXAMPLE",
            "fit_score": 0.85,
            "hired": 1,  # Fair = hired
            "bias_detected": 0,
            "bias_type": "none",
            "bias_reason": "Fair decision - no bias",
            "is_synthetic": 1
        })

        # Biased decision
        training_data.append({
            "resume_id": f"BIAS_BIASED_{i}",
            "job_id": example["job_id"],
            "job_title": job_title,
            "resume_text": example["resume_text"],
            "job_description": job_description,
            "resume_category": "BIAS_EXAMPLE",
            "fit_score": 0.85,
            "hired": 0,  # Biased = rejected
            "bias_detected": 1,
            "bias_type": example["bias_type"],
            "bias_reason": example["bias_reason"],
            "is_synthetic": 1
        })

    df = pd.DataFrame(training_data)

    print(f"\n✅ Total training samples: {len(df)}")
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

    # Balance check
    print("\n📈 BALANCE CHECK:")
    hire_rate = df['hired'].mean()
    if 0.45 <= hire_rate <= 0.65:
        print(f"  ✅ BALANCED: {hire_rate*100:.1f}% hire rate")
    else:
        print(f"  ⚠️ IMBALANCED: {hire_rate*100:.1f}% hire rate (target: 45-65%)")

    # Category bias check
    print("\n📈 CATEGORY FAIRNESS CHECK:")
    tech_cats = ['INFORMATION-TECHNOLOGY', 'ENGINEERING', 'DATA-SCIENCE']
    non_tech_cats = ['CHEF', 'ARTS', 'FITNESS', 'TEACHER']

    tech_hire = df[df['resume_category'].isin(tech_cats)]['hired'].mean()
    non_tech_hire = df[df['resume_category'].isin(non_tech_cats)]['hired'].mean()

    print(f"  Tech hire rate: {tech_hire*100:.1f}%")
    print(f"  Non-tech hire rate: {non_tech_hire*100:.1f}%")

    if abs(tech_hire - non_tech_hire) > 0.30:
        print(f"  ⚠️ LARGE GAP: {abs(tech_hire - non_tech_hire)*100:.1f}% difference")
    else:
        print(f"  ✅ FAIR: {abs(tech_hire - non_tech_hire)*100:.1f}% difference")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("="*50)
    print("🚀 FAIRHIRE AI - DATA PROCESSING PIPELINE (BALANCED)")
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