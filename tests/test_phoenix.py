"""
Test Arize Phoenix setup
"""
import phoenix as px

print("Starting Phoenix server...")
session = px.launch_app()

print(f"✅ Phoenix running at: {session.url}")
print(f"✅ Open browser: http://localhost:6006")
print("\nPress Ctrl+C to stop...")

# Keep running
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nPhoenix stopped.")