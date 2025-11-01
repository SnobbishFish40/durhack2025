"""
Test if environment variables are loading correctly
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("🔍 Testing Environment Variables\n")
print("="*50)

# Check NASA key
nasa_key = os.getenv('NASA_API_KEY', 'NOT_FOUND')
print(f"NASA_API_KEY: {nasa_key[:20]}..." if len(nasa_key) > 20 else f"NASA_API_KEY: {nasa_key}")

if nasa_key == 'DEMO_KEY':
    print("  ✅ Using NASA DEMO_KEY (public key)")
elif nasa_key == 'NOT_FOUND':
    print("  ❌ NASA key not found in .env")
else:
    print("  ✅ Custom NASA key loaded")

print()

# Check Anthropic key
anthropic_key = os.getenv('ANTHROPIC_API_KEY', 'NOT_FOUND')
if anthropic_key == 'NOT_FOUND':
    print("ANTHROPIC_API_KEY: NOT_FOUND")
    print("  ⚠️  Anthropic key not set (optional)")
else:
    print(f"ANTHROPIC_API_KEY: {anthropic_key[:20]}...")
    print("  ✅ Anthropic key loaded")

print("="*50)
