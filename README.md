# FairHire AI

## An AI Agent for Fair Resume Screening

FairHire AI is an intelligent monitoring agent that watches resume screening models 24/7, detects bias and drift in real-time, and explains findings in plain English to help HR teams make fair hiring decisions.

### Problem Statement

Companies deploy AI resume screeners to hire at scale. But these models learn human biases from historical hiring data:
- Women get rejected at higher rates
- Non-traditional educational backgrounds penalized
- Nationality and ethnicity influence decisions
- Nobody notices until legal risk surfaces

**Real Example:** Amazon's AI hiring tool was scrapped in 2018 because it discriminated against women. The problem still exists across thousands of companies today.

### Solution: FairHire AI

An AI agent powered by Gemini that monitors resume screening models and alerts HR teams when bias emerges.

**What It Does:**
1. ✅ Monitors model accuracy 24/7
2. ✅ Detects when models become biased
3. ✅ Explains why in plain English (using SHAP/LIME)
4. ✅ Recommends corrective actions
5. ✅ Runs automatically on Arize MCP

### Key Features

- **Real-time Bias Detection:** Monitors hiring rates across demographic groups
- **Explainable AI:** Uses SHAP and LIME to explain decisions
- **Multi-step Agent:** Intelligent agent that investigates issues independently
- **Interactive Dashboard:** Built with Streamlit for HR managers
- **Production-Ready:** Uses industry tools (Gemini, Arize, Google Cloud)

### How It Works
HR Manager Asks: "Why are we rejecting women?"
↓
Agent investigates via Arize MCP
↓
Finds: Women 60% rejection, Men 28% rejection
↓
Root cause: Training data was 78% male
↓
SHAP explains which features caused bias
↓
Agent returns:
"Your model has learned gender bias.
Recommendation: Retrain with balanced data.
Legal risk: High. Audit last 100 rejections."

### Tech Stack

**AI & ML**
- PyTorch / TensorFlow
- BERT Transformers
- SHAP / LIME (Explainability)

**Cloud & Infrastructure**
- Google Cloud Platform
- Google Cloud Agent Builder
- Gemini 1.5 Pro LLM
- Arize (Model Monitoring via MCP)

**Data & Dashboard**
- Pandas / NumPy / Scikit-learn
- Streamlit (Dashboard)
- Plotly (Visualizations)
- SQLite / Turso Cloud (Database)

### Getting Started

#### Prerequisites
- Python 3.11+
- Google Cloud Account ($300 free credits)
- Arize Account (free tier available)
- GitHub Account

#### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/LaraibKaleem/fairhire-ai
   cd fairhire-ai
```

2. **Create virtual environment**
```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate   # Mac/Linux
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up credentials**
```bash
   # Create .env file with:
   GCP_PROJECT_ID=your-project-id
   ARIZE_ORG_KEY=your-org-key
   ARIZE_API_KEY=your-api-key
```

5. **Run dashboard**
```bash
   streamlit run dashboard/app.py
```

### Project Structure
fairhire-ai/
├── agent/                    # Agent logic & MCP integration
├── data/                    # Datasets (resumes, job descriptions)
├── dashboard/               # Streamlit dashboard
├── models/                 # Trained BERT model
├── notebooks/              # Jupyter notebooks for experimentation
├── tests/                  # Unit tests
├── xai/                    # SHAP & LIME explainability
├── ARCHITECTURE.md         # Detailed system design
├── LICENSE                 # MIT License
├── README.md              # This file
└── requirements.txt       # Python dependencies

### Usage

#### Run Dashboard
```bash
streamlit run dashboard/app.py
```

#### Train Model (Phase 2)
```bash
jupyter notebook notebooks/02_bert_training.ipynb
```

#### Run Tests
```bash
pytest tests/
```

### Phases & Timeline

| Phase | Days | Focus | Status |
|-------|------|-------|--------|
| 1 | 1-3 | Setup & Documentation | ✅ Complete |
| 2 | 4-5 | Data Collection | ✅ Complete |
| 3 | 6-10 | Model Training | ✅ Complete |
| 4 | 11-14 | Arize Integration | ✅ Complete |
| 5 | 15-22 | Agent Development | ✅ Complete |
| 6 | 23-25 | Dashboard & Deployment | ✅ Complete |
| 7 | 26-28 | Demo & Submission | ⏳ Pending |

### Business Impact

- **Legal Protection:** Prevents discrimination lawsuits
- **Compliance:** Meets EU AI Act and local regulations
- **Hiring Quality:** Recognizes talent beyond keywords
- **Diversity:** Increases representation across all backgrounds
- **Reputation:** Protects brand from public backlash

### Author

**Laraib Kaleem**
📧 Email: laraibkaleem15@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/laraibkaleem/
🐙 GitHub: https://github.com/LaraibKaleem

### Hackathon

**Google Cloud Rapid Agent Hackathon 2026**
- **Track:** Arize
- **Deadline:** June 12, 2026
<!-- - **Prize:** $5,000 (1st place) -->

### License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

### Acknowledgments

- Google Cloud for providing $300 free credits
- Arize for MCP server integration
- Gemini for advanced reasoning capabilities
- Hugging Face for BERT models

<!-- ### Contact & Support

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Email: laraibkaleem15@gmail.com
- Discord: [Community Discord Link]

--- -->

**Built with ❤️ for fair hiring powered by AI**
