"""
check_models.py
---------------
Lists all Gemini models available for the configured API key.
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.config import settings
from google import genai

client = genai.Client(api_key=settings.google_api_key)

print("Available Gemini models that support generateContent:\n")
for m in client.models.list():
    methods = getattr(m, "supported_generation_methods", []) or []
    if not methods or "generateContent" in methods:
        print(f"  {m.name}")
