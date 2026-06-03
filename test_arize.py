import os
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

org_key = os.getenv("ARIZE_ORG_KEY")
api_key = os.getenv("ARIZE_API_KEY")
project_id = os.getenv("GCP_PROJECT_ID")

print("Testing Arize Connection...")
print(f"✅ Organization Key loaded: {org_key[:10]}..." if org_key else "❌ Org Key missing")
print(f"✅ API Key loaded: {api_key[:10]}..." if api_key else "❌ API Key missing")
print(f"✅ GCP Project ID loaded: {project_id}" if project_id else "❌ Project ID missing")

if org_key and api_key and project_id:
    print("\n🎉 All credentials loaded successfully!")
    print("Ready for Phase 2: Data Collection")
else:
    print("\n❌ Some credentials missing. Check .env file.")