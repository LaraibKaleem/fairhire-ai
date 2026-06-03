# FairHire AI - System Architecture

## Overview
FairHire AI is a multi-component system that monitors resume screening models for bias and drift in real-time using AI agents.

## System Components

### 1. Resume Screener (BERT Model)
- **Technology:** BERT Transformer (Hugging Face)
- **Framework:** PyTorch
- **Input:** Resume text + Job description
- **Output:** Hiring score (0-1 probability)
- **Performance Target:** 85%+ accuracy
- **Location:** `models/bert_screener/`

### 2. Data Pipeline
- **Input:** Raw resumes (PDF, DOCX, TXT)
- **Processing:** Text extraction, tokenization, cleaning
- **Output:** Training dataset (CSV)
- **Location:** `data/`
  - `data/resumes_raw/` - Original resumes
  - `data/resumes_processed/` - Cleaned data
  - `training_data.csv` - Final training set (2000+ samples)

### 3. Arize Monitoring
- **Service:** Arize MCP Server
- **Purpose:** Monitor model performance, detect drift
- **Tracks:**
  - Model accuracy
  - Precision, Recall, F1-Score
  - Data quality metrics
  - Bias across demographic groups (gender, ethnicity, education)
- **Alerts:** Triggered when thresholds exceeded
- **MCP Integration:** Agent connects via MCP protocol

### 4. Gemini Agent (Monitoring Brain)
- **LLM:** Google Gemini 1.5 Pro
- **Platform:** Google Cloud Agent Builder
- **Capabilities:**
  - Receives alerts from Arize
  - Investigates model performance issues
  - Reasons about root causes
  - Generates plain English explanations
  - Recommends corrective actions
- **Workflow:** Multi-step investigation process

### 5. XAI Layer (Explainability)
- **SHAP:** Feature importance (deep mathematical rigor)
- **LIME:** Local interpretable explanations (fast, user-friendly)
- **Output:** Plain English explanations of why candidates were hired/rejected
- **Location:** `xai/`

### 6. Streamlit Dashboard
- **Technology:** Streamlit
- **Hosting:** Streamlit Cloud
- **Features:**
  - Real-time model metrics display
  - Interactive investigation tool
  - Bias analytics by demographic group
  - Alert management
  - 7-day trend analysis
- **Location:** `dashboard/app.py`

## Data Flow
Raw Resumes (2000+)
↓
Data Processing (Pandas, NumPy)
↓
Training Dataset (CSV)
↓
BERT Fine-tuning (PyTorch)
↓
Trained Model (85%+ accuracy)
↓
Arize Upload
↓
Monitoring Active (24/7)
↓
Drift Detected → Alert Triggered
↓
Agent Investigates (Gemini)
↓
SHAP Explains Features
↓
Agent Reasons & Explains
↓
Streamlit Dashboard Shows Result
↓
HR Manager Gets Plain English Explanation

## Technology Stack

### Core ML/AI
- **PyTorch:** Deep learning framework
- **Transformers (Hugging Face):** BERT model
- **Scikit-learn:** ML utilities
- **TensorFlow:** Alternative ML framework

### Monitoring & Explainability
- **Arize:** Model monitoring via MCP
- **SHAP:** Feature importance
- **LIME:** Local explanations

### Cloud & Infrastructure
- **Google Cloud Platform:** Main cloud provider
- **Google Cloud Agent Builder:** Agent platform
- **Gemini API:** LLM brain
- **Streamlit Cloud:** Dashboard hosting
- **Turso Cloud:** Database (SQLite edge)

### Data Processing
- **Pandas:** Data manipulation
- **NumPy:** Numerical computing
- **Scikit-learn:** Preprocessing

### Dashboard & Visualization
- **Streamlit:** UI framework
- **Plotly:** Interactive charts

### Development
- **Python 3.11:** Language
- **Git/GitHub:** Version control
- **Jupyter:** Experimentation notebooks
- **VS Code:** IDE

## Deployment Architecture
┌─────────────────────────────────────────┐
│         End User (HR Manager)           │
│  "Why are we rejecting women?"          │
└──────────────────┬──────────────────────┘
│
┌──────────▼──────────┐
│ Streamlit Dashboard │
│ (Streamlit Cloud)   │
└──────────┬──────────┘
│
┌──────────▼──────────────────┐
│  Google Cloud Agent Builder  │
│  - Gemini 1.5 Pro            │
│  - Agent Logic               │
└──────────┬───────────────────┘
│
┌──────────▼──────────┐
│  Arize MCP Server   │
│  - Model Monitoring │
│  - Drift Detection  │
│  - Bias Metrics     │
└──────────┬──────────┘
│
┌──────────▼──────────────┐
│  Resume Screener (BERT) │
│  - Scoring Resume       │
│  - Making Predictions   │
└──────────┬──────────────┘
│
┌──────────▼──────────┐
│ SQLite Database     │
│ (Turso Cloud)       │
│ - Training Data     │
│ - Predictions       │
└─────────────────────┘

## Key Features

### 1. Real-time Monitoring
- 24/7 monitoring via Arize
- Instant alerts when drift detected
- Automatic investigation by agent

### 2. Bias Detection
- Tracks hiring rate by demographic group
- Detects statistical disparities
- Calculates fairness metrics

### 3. Explainability
- SHAP shows feature importance
- LIME provides fast local explanations
- Gemini generates plain English reports

### 4. Multi-step Agent
- Receives user questions
- Calls Arize tools to gather data
- Reasons with Gemini
- Returns actionable recommendations

### 5. Dashboard
- Visualizes model health
- Interactive investigation tool
- Bias analytics
- Alert management

## Performance Metrics

### Model Performance
- **Accuracy Target:** 85%+
- **Precision Target:** 85%+
- **Recall Target:** 85%+
- **F1 Score Target:** 85%+

### Fairness Metrics
- **Gender Parity:** <10% difference acceptable
- **Ethnicity Parity:** <10% difference acceptable
- **Education Parity:** <10% difference acceptable

### Monitoring
- **Drift Detection:** Daily checks
- **Alert Response:** Real-time
- **Data Quality:** 95%+ threshold

## Project Timeline

- **Phase 1 (Days 1-3):** Setup & Documentation ✅
- **Phase 2 (Days 4-5):** Data Collection
- **Phase 3 (Days 6-10):** Model Training
- **Phase 4 (Days 11-14):** Arize Integration
- **Phase 5 (Days 15-22):** Agent Development
- **Phase 6 (Days 23-25):** Dashboard Deployment
- **Phase 7 (Days 26-28):** Demo & Submission

## Files & Directories
fairhire-ai/
├── .venv/                    # Virtual environment
├── agent/                    # Agent logic
│   ├── prompts/             # Gemini prompts
│   └── tools/               # Arize MCP integration
├── data/                    # Datasets
│   ├── resumes_raw/         # Raw resume files
│   └── resumes_processed/   # Cleaned data
├── dashboard/               # Streamlit app
│   └── app.py              # Main dashboard
├── models/                 # Trained models
│   └── bert_screener/      # BERT model artifacts
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_bert_training.ipynb
│   └── 03_arize_setup.ipynb
├── tests/                  # Unit tests
├── xai/                    # Explainability
│   ├── shap_explainer.py
│   └── lime_explainer.py
├── .env                    # Credentials (local only)
├── .gitignore             # Git ignore rules
├── ARCHITECTURE.md        # This file
├── LICENSE                # MIT License
├── README.md              # Project overview
└── requirements.txt       # Python dependencies

## Future Extensions

### Phase 2 (Post-hackathon)
- Resume parser for PDF/DOCX
- Applicant database
- Interview scheduling
- Offer management
- Full ATS system

### Phase 3 (Productization)
- Multilingual support (Urdu, Spanish, etc.)
- Advanced fairness constraints
- Custom model training
- Enterprise integrations
- White-label SaaS offering
