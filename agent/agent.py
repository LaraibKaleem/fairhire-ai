"""
FairHire AI - Main Monitoring Agent
Powered by Gemini + Arize Phoenix
"""

import os
import json
import sys
import time
from dotenv import load_dotenv
from google import genai
from phoenix.otel import register
import phoenix as px
from opentelemetry import trace

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from tools.model_tools import (
    screen_resume,
    explain_decision,
    detect_bias,
    get_model_metrics
)

load_dotenv()

# ============================================================
# SETUP PHOENIX
# ============================================================

def setup_phoenix():
    """Connect agent to Phoenix for tracing"""
    
    print("\n🔍 Connecting to Arize Phoenix...")
    
    try:
        tracer_provider = register(
            project_name="fairhire-ai",
            endpoint="http://localhost:6006/v1/traces",
            auto_instrument=True
        )
        print("✅ Phoenix connected: http://localhost:6006")
        return tracer_provider
        
    except Exception as e:
        print(f"⚠️ Phoenix: {e}")
        return None

# ============================================================
# SETUP GEMINI
# ============================================================

def setup_gemini():
    """Initialize Gemini model"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")
    
    client = genai.Client(api_key=api_key)
    
    prompt_path = "agent/prompts/system_prompt.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            system_prompt = f.read()
    else:
        system_prompt = "You are FairHire AI monitoring agent."
    
    print("✅ Gemini 2.0 Flash initialized")
    return client, system_prompt

# ============================================================
# AGENT CLASS
# ============================================================

class FairHireAgent:
    
    def __init__(self):
        self.client, self.system_prompt = setup_gemini()
        self.tracer = trace.get_tracer("fairhire-ai")
        print("✅ FairHire Agent ready!")
    
    def call_gemini(self, prompt_text: str) -> str:
        """Call Gemini with Phoenix manual tracing"""
        
        # Manual Phoenix span
        with self.tracer.start_as_current_span(
            "gemini.generate_content"
        ) as span:
            span.set_attribute("input.value", prompt_text[:500])
            span.set_attribute("llm.model", "gemini-2.0-flash")
            
            try:
                time.sleep(3)
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt_text,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=self.system_prompt
                    )
                )
                result = response.text
                span.set_attribute("output.value", result[:500])
                span.set_attribute("llm.status", "success")
                return result
                
            except Exception as e:
                span.set_attribute("llm.status", "quota_exceeded")
                span.set_attribute("llm.error", str(e))
                return None
    
    def investigate(
        self,
        resume_text: str,
        job_title: str,
        candidate_name: str = "Candidate"
    ) -> dict:
        """Full candidate investigation"""
        
        # Phoenix span for full investigation
        with self.tracer.start_as_current_span(
            f"investigate.{candidate_name}"
        ) as span:
            
            span.set_attribute("candidate.name", candidate_name)
            span.set_attribute("candidate.job", job_title)
            
            print(f"\n{'='*50}")
            print(f"🔍 Investigating: {candidate_name}")
            print(f"📋 Job: {job_title}")
            print(f"{'='*50}")
            
            # Step 1: Screen
            print("\n📊 Step 1: Screening resume...")
            screen = screen_resume(resume_text, job_title)
            print(f"   Decision: {screen['decision']}")
            print(f"   Probability: {screen['hire_percentage']}%")
            span.set_attribute(
                "screening.decision", screen['decision']
            )
            span.set_attribute(
                "screening.probability", 
                screen['hire_percentage']
            )
            
            # Step 2: Explain
            print("\n🔬 Step 2: SHAP explanation...")
            explanation = explain_decision(resume_text, job_title)
            if "error" not in explanation:
                print(f"   Top positive: "
                      f"{explanation.get('top_positive', 'none')}")
                print(f"   Top negative: "
                      f"{explanation.get('top_negative', 'none')}")
                span.set_attribute(
                    "shap.top_positive",
                    explanation.get('top_positive', 'none')
                )
                span.set_attribute(
                    "shap.top_negative",
                    explanation.get('top_negative', 'none')
                )
            
            # Step 3: Bias check
            print("\n⚖️ Step 3: Bias detection...")
            bias = detect_bias(
                resume_text,
                screen['decision'],
                screen['hire_probability']
            )
            print(f"   Bias detected: {bias['bias_detected']}")
            if bias['bias_flags']:
                print(f"   Flags: {', '.join(bias['bias_flags'])}")
            span.set_attribute(
                "bias.detected", bias['bias_detected']
            )
            span.set_attribute(
                "bias.flags",
                str(bias['bias_flags'])
            )
            
            # Step 4: Gemini reasoning
            print("\n🤖 Step 4: Gemini analysis...")
            
            # PROMPT DEFINED HERE (not commented out)
            prompt = f"""
Analyze this hiring decision:

CANDIDATE: {candidate_name}
JOB: {job_title}
MODEL DECISION: {screen['decision']}
HIRE PROBABILITY: {screen['hire_percentage']}%

TOP POSITIVE FACTORS (support hiring):
{json.dumps(explanation.get('positive_features', []), indent=2)}

TOP NEGATIVE FACTORS (against hiring):
{json.dumps(explanation.get('negative_features', []), indent=2)}

BIAS ANALYSIS:
Bias detected: {bias['bias_detected']}
Bias flags: {', '.join(bias['bias_flags']) if bias['bias_flags'] else 'None'}

CANDIDATE RESUME SUMMARY:
{resume_text[:400]}

Provide analysis with these 4 sections:
DECISION SUMMARY:
KEY FACTORS:
BIAS ANALYSIS:
RECOMMENDATION:
"""
            
            # Call Gemini (with fallback)
            gemini_analysis = self.call_gemini(prompt)
            
            if gemini_analysis is None:
                print("   ⚠️ Using fallback analysis...")
                gemini_analysis = f"""
DECISION SUMMARY:
Model decision: {screen['decision']} with {screen['hire_percentage']}% probability for {candidate_name}.

KEY FACTORS:
Positive: {explanation.get('top_positive', 'relevant skills detected')}
Negative: {explanation.get('top_negative', 'some keywords missing')}

BIAS ANALYSIS:
Bias detected: {bias['bias_detected']}
Flags: {', '.join(bias['bias_flags']) if bias['bias_flags'] else 'None detected'}
{bias['recommendation']}

RECOMMENDATION:
{'⚠️ Manual review required due to bias flags detected.' if bias['bias_detected'] else '✅ Proceed with standard hiring process.'}
"""
            
            print("   ✅ Analysis complete")
            span.set_attribute(
                "gemini.analysis", gemini_analysis[:300]
            )
            
            print("\n" + "="*50)
            print("📋 ANALYSIS RESULT")
            print("="*50)
            print(gemini_analysis)
            
            result = {
                "candidate": candidate_name,
                "job": job_title,
                "decision": screen['decision'],
                "probability": screen['hire_percentage'],
                "bias_detected": bias['bias_detected'],
                "bias_flags": bias['bias_flags'],
                "final_decision": bias['final_decision'],
                "gemini_analysis": gemini_analysis
            }
            
            return result
    
    def answer_question(self, question: str) -> str:
        """Answer HR manager question"""
        
        print(f"\n❓ Question: {question}")
        
        with self.tracer.start_as_current_span(
            "answer_question"
        ) as span:
            span.set_attribute("question", question)
            
            metrics = get_model_metrics()
            
            prompt = f"""
HR Manager asks: {question}

Current Model Status:
- Accuracy: {metrics.get('accuracy')}%
- Precision: {metrics.get('precision')}%
- Recall: {metrics.get('recall')}%
- F1: {metrics.get('f1_score')}%
- Status: {metrics.get('status')}
- Model: {metrics.get('model_type')}

Answer professionally and clearly.
"""
            
            answer = self.call_gemini(prompt)
            
            if answer is None:
                answer = f"""
Model Status Report:
- Accuracy: {metrics.get('accuracy')}%
- Precision: {metrics.get('precision')}%
- Status: {metrics.get('status')}
- Bias monitoring is active.
- Model performing adequately for fair screening.
"""
            
            span.set_attribute("answer", answer[:300])
            print(f"\n🤖 {answer}")
            return answer
    
    def check_health(self) -> dict:
        """Check model health"""
        
        with self.tracer.start_as_current_span(
            "check_health"
        ) as span:
            
            metrics = get_model_metrics()
            accuracy = metrics.get('accuracy', 0)
            
            status = (
                "EXCELLENT ✅" if accuracy >= 85 else
                "GOOD ✅" if accuracy >= 75 else
                "FAIR ⚠️" if accuracy >= 65 else
                "POOR ❌"
            )
            
            span.set_attribute("health.status", status)
            span.set_attribute("health.accuracy", accuracy)
            
            print(f"\n📊 Model Health: {status}")
            print(f"   Accuracy: {accuracy}%")
            
            return {
                "status": status,
                "accuracy": accuracy,
                "metrics": metrics
            }

# ============================================================
# RUN TESTS
# ============================================================

def run_tests():
    """Run agent tests"""
    
    print("\n" + "="*60)
    print("🚀 FAIRHIRE AI - AGENT TESTS")
    print("="*60)
    
    agent = FairHireAgent()
    
    # Test 1: Health check
    print("\n📊 TEST 1: Model Health")
    health = agent.check_health()
    
    # Test 2: Qualified candidate
    print("\n🧪 TEST 2: Qualified Candidate")
    r1 = agent.investigate(
        resume_text="""Laraib Kaleem AI ML Engineer
        MS Data Science Bahria University Pakistan
        Published NLP researcher Springer Nature 2024
        BERT T5 Transformers PyTorch TensorFlow SHAP
        Samsung Innovation Campus AI Trainee 2026
        machine learning deep learning python
        data science research publications NLP""",
        job_title="Senior ML Engineer",
        candidate_name="Laraib Kaleem"
    )
    
    # Test 3: Bias case
    print("\n🧪 TEST 3: Bias Case")
    r2 = agent.investigate(
        resume_text="""Fatima Khan Senior Data Scientist
        PhD Computer Science MIT 8 years experience
        Published 12 papers top ML conferences
        Expert NLP deep learning Python TensorFlow
        Led teams Amazon machine learning research""",
        job_title="Senior ML Engineer",
        candidate_name="Fatima Khan"
    )
    
    # Test 4: HR Question
    print("\n🧪 TEST 4: HR Manager Question")
    answer = agent.answer_question(
        "Is our screening model fair to all candidates?"
    )
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE!")
    print("="*60)
    print("📌 Check Phoenix: http://localhost:6006")
    
    return {
        'health': health,
        'test1': r1,
        'test2': r2,
        'answer': answer
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    # Start Phoenix
    phoenix_session = setup_phoenix()
    
    # Run tests
    results = run_tests()
    
    print("\n🎉 Day 6 Complete!")
    print("🔍 Phoenix: http://localhost:6006")
    print("📌 Next: Streamlit Dashboard")