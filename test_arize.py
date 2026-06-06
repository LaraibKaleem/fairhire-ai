import os
from dotenv import load_dotenv
from phoenix.otel import register

# Load credentials from .env
load_dotenv()

# Google Cloud
project_id = os.getenv("GCP_PROJECT_ID")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Phoenix Local (NO API KEY NEEDED)
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")

# Connect to local Phoenix
tracer_provider = register(
    project_name="my-hackathon-agent",
    endpoint=PHOENIX_ENDPOINT,
    auto_instrument=True
)

print("=" * 50)
print("🚀 Hackathon Agent Setup")
print("=" * 50)

# Check Google Cloud
if project_id:
    print(f"✅ GCP Project ID: {project_id}")
else:
    print("❌ GCP Project ID missing — add to .env")

if google_api_key:
    print(f"✅ Google API Key: {google_api_key[:10]}...")
else:
    print("❌ GOOGLE_API_KEY missing — add to .env")

# Check Phoenix
print(f"✅ Phoenix Endpoint: {PHOENIX_ENDPOINT}")

print("=" * 50)

if project_id and google_api_key:
    print("🎉 Ready to build your agent!")
    print("   - Gemini 3 will use GOOGLE_API_KEY")
    print("   - Traces will send to Phoenix at", PHOENIX_ENDPOINT)
else:
    print("❌ Fix missing credentials in .env file")