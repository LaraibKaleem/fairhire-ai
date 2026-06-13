# ⚖️ FairHire AI

###  Bias-Free Recruitment Intelligence Systems

FairHire AI is an intelligent monitoring agent that audits AI-powered resume screening systems for bias and fairness in real time. It screens candidates, explains every decision using SHAP (SHapley Additive exPlanations), detects discrimination patterns across six bias categories, and delivers plain-English recommendations to HR managers — with every agent step fully traced via Arize Phoenix.

---

## 🚨 The Problem

Companies deploy AI resume screeners to hire at scale. But these models silently learn human biases from historical hiring data:

- Women get rejected at higher rates than equally qualified men
- Candidates from non-prestige universities are penalized unfairly
- Nationality, ethnicity, and name origin influence decisions
- Career gaps (e.g., maternity leave) trigger automatic rejection
- Nobody notices until a lawsuit surfaces

**Real Example:** Amazon scrapped its AI hiring tool in 2018 after discovering it systematically discriminated against women. The same problem exists across thousands of companies today — silently.

**Our Discovery:** After training FairHire AI's own screening model on real resume data, SHAP revealed:
- `"minority"` → SHAP impact of **-0.0165** (model penalizes minority candidates)
- `"local institute"` → SHAP impact of **-0.0149** (education bias detected)
- `"ai"` → **negative** impact even for an AI/ML engineering role (illogical learned correlation)

This is not a hypothetical problem. It is measurable and provable — and FairHire AI proves it.

---

## ✅ The Solution

FairHire AI acts as an automated, transparent auditor placed on top of any resume screening pipeline. It does not replace the existing model — it watches it, explains it, and flags when it discriminates.

### Four-Step Pipeline
**Step 1: SCREEN**
- Random Forest + TF-IDF classifier
- screens the resume → HIRE / REJECT + probability score

**Step 2: EXPLAIN (SHAP[SHapley Additive exPlanations identifies])**
- exactly which words drove the decision (positive factors vs. negative factors)

**Step 3: DETECT BIAS**
- Rule-based engine checks across
- 6 bias categories → flags discrimination

**Step 4: REASON (Gemini)**
- Gemini 2.0 Flash synthesizes all findings
- into plain-English summary + recommendation

**for the HR manager**
- All steps traced by Arize Phoenix (OpenTelemetry)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📤 Resume Screening | Upload PDF/DOCX/TXT or paste text, select job position, get instant HIRE/REJECT decision |
| 🔬 SHAP Explainability | Visual breakdown of which resume words/phrases helped or hurt the candidate |
| ⚖️ Bias Detection | Six-category bias risk scoring with radar chart, bar chart, and toggle filters |
| 🤖 Gemini Reasoning | Plain-English decision summary, key factors, bias analysis, and HR recommendation |
| 👥 Similar Candidates | Skill-overlap matching (Jaccard similarity) to find comparable candidates |
| 📈 Bias Analytics | Trend charts showing bias risk patterns over time across all screened candidates |
| 📋 Screening History | Full log of all decisions with CSV export for audit/compliance |
| 🔗 Phoenix Observability | Every agent step traced via Arize Phoenix + OpenTelemetry |

---

## 🛠️ Tech Stack

### AI & Machine Learning
| Tool | Full Form | Role |
|---|---|---|
| Random Forest | Random Forest (ensemble of decision trees) | Core resume screening model |
| TF-IDF | Term Frequency – Inverse Document Frequency | Converts resume text to numerical features |
| SHAP | SHapley Additive exPlanations | Explains individual model predictions |
| LIME | Local Interpretable Model-agnostic Explanations | Secondary explainability method |
| scikit-learn | Scientific Kit – Learn (Python ML library) | Implements RF classifier and TF-IDF vectorizer |

### LLM & Agent
| Tool | Full Form | Role |
|---|---|---|
| Gemini 2.0 Flash | Google Gemini 2.0 Flash (Large Language Model) | Plain-English reasoning and recommendations |
| Google GenAI SDK | Google Generative AI Software Development Kit | Python client for Gemini API calls |
| Google Cloud Agent Builder | Google Cloud Agent Builder (GCP AI platform) | Cloud infrastructure for agent deployment |

### Observability
| Tool | Full Form | Role |
|---|---|---|
| Arize Phoenix | Arize AI Phoenix (open-source LLM observability) | Traces every agent step at localhost:6006 |
| OpenTelemetry | Open Telemetry (open standard for observability) | Underlying tracing protocol and span management |
| OTel SDK | OpenTelemetry Software Development Kit | Python SDK for creating and exporting spans |

### Dashboard & Data
| Tool | Full Form | Role |
|---|---|---|
| Streamlit | Streamlit (Python web app framework) | Interactive HR manager dashboard |
| Plotly | Plotly (interactive graphing library) | Radar charts, gauges, bar charts, trend lines |
| Pandas | Panel Data Analysis library (Python) | Data manipulation and dataset operations |
| NumPy | Numerical Python | Array operations and numerical computing |

### Infrastructure
| Tool | Full Form | Role |
|---|---|---|
| GCP | Google Cloud Platform | Cloud project and API management |
| GitHub | Git-based code hosting platform | Source code, version control, public repo |
| Streamlit Community Cloud | Streamlit Community Cloud (free hosting) | Public deployment for judge access |
| Python-dotenv | Python dotenv (environment variable loader) | Manages API keys via .env file |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | **81.85%** |
| Precision | **90.78%** |
| Recall | **70.83%** |
| F1 Score | **79.57%** |
| Training Samples | 2,116 |
| Test Samples | 529 |
| Total Dataset | 2,645 resumes (50% hired / 50% rejected) |

---

## 🚀 Getting Started

### Prerequisites
Python 3.11+

Google Cloud Account (free $300 credits available)

Gemini API Key (free tier at aistudio.google.com)

GitHub Account

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/LaraibKaleem/fairhire-ai
cd fairhire-ai
```

**2. Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
source .venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up credentials**
```bash
# Create .env file with:
GCP_PROJECT_ID=your-project-id
GOOGLE_API_KEY=your-gemini-api-key
PHOENIX_PORT=6006
```

**5. Run the dashboard**
```bash
streamlit run dashboard/app.py
```

**6. (Optional) Run agent with Phoenix tracing**
```bash
# Terminal 1 — start Phoenix server
python -m phoenix.server.main serve

# Terminal 2 — run agent
python agent/agent.py
```

**7. View Phoenix traces**
Open browser: http://localhost:6006

Click: Traces tab

See every agent step traced in real time

---

## 💼 Business Impact

| Benefit | Detail |
|---|---|
| ⚖️ Legal Protection | Prevents discrimination lawsuits before they happen |
| 📋 Compliance | Supports EU AI Act, EEOC (US), Equality Act (UK) audit requirements |
| 🎯 Hiring Quality | Surfaces qualified candidates that keyword-matching misses |
| 🌍 Diversity | Increases representation by catching demographic bias early |
| 🏢 Reputation | Protects employer brand from public discrimination backlash |
| 💰 Cost Saving | Catching bias early is far cheaper than defending a lawsuit |

---

## 🐛 Known Issues & Limitations

- Phoenix observability runs locally only (not available on Streamlit Cloud deployment)
- Gemini free tier limited to 20 requests/day on gemini-2.0-flash; fallback analysis used when quota exceeded
- Bias detection is rule-based for MVP; future versions will use statistical fairness metrics (disparate impact ratio, equal opportunity difference)
- LIME integration installed but not yet live in the pipeline (planned for next version)

---

## 🔮 Future Work

- Bias-corrected model retraining using SHAP findings
- Cloud-hosted Arize Phoenix instance
- ATS system connectors (Workday, Greenhouse, Lever)
- Expanded bias categories (disability, veteran status, religion)
- Statistical fairness metrics (disparate impact, equal opportunity)
- Automated compliance report generation (PDF)
- Multi-company SaaS deployment with authentication

---

## 🙏 Acknowledgments

- Google Cloud — $300 free credits for GCP project setup
- Arize AI — Phoenix open-source observability platform
- Google Gemini — advanced LLM reasoning capabilities
- Kaggle (snehaanbhawal) — Resume dataset (2,485 resumes, 25 categories)
- Anthropic Claude — development assistance throughout the 9-day sprint

---

*⚖️ Built because every candidate deserves a fair chance.*
