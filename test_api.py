#!/usr/bin/env python3
# ============================================================
# test_api.py
# Standalone test to verify your Google API key works and
# identify the correct model name string for Gemma.
#
# Place this file anywhere (does NOT need to be inside grocery_agent/).
# Run with: python test_api.py
# or:        uv run python test_api.py
# ============================================================

import os
import sys
from dotenv import load_dotenv

# Load .env from the same directory, or from grocery_agent/ if present
load_dotenv()
load_dotenv("grocery_agent/.env")

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY or API_KEY == "your_google_api_key_here":
    print("ERROR: GOOGLE_API_KEY not set in .env file.")
    sys.exit(1)

print(f"API key loaded: {API_KEY[:8]}...{API_KEY[-4:]}\n")

# ── Test with new SDK (google-genai) ────────────────────────
print("=" * 55)
print("Testing with new SDK: google-genai")
print("=" * 55)

try:
    from google import genai

    client = genai.Client(api_key=API_KEY)

    # List of candidate model names to try
    candidates = [
        "gemma-3-27b-it",
        "models/gemma-3-27b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ]

    for model_name in candidates:
        print(f"\n  Trying model: {model_name} ...", end=" ", flush=True)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Reply with exactly 5 words: 'Grocery agent is working fine.'",
            )
            print(f"OK")
            print(f"  Response: {response.text.strip()}")
            print(f"\n  --> Use this model name in settings.py: \"{model_name}\"")
            break  # stop at first success
        except Exception as e:
            err = str(e)
            if "not found" in err.lower() or "invalid" in err.lower() or "404" in err:
                print(f"FAIL (model not found)")
            elif "429" in err or "exhausted" in err.lower():
                print(f"FAIL (rate limit / quota: {err[:80]})")
            elif "401" in err or "api key" in err.lower():
                print(f"FAIL (bad API key)")
                break
            else:
                print(f"FAIL ({err[:100]})")

except ImportError:
    print("  google-genai package not installed. Run: uv add google-genai")

# ── Also list all available models ──────────────────────────
print("\n" + "=" * 55)
print("Listing available Gemma models on your account:")
print("=" * 55)

try:
    from google import genai
    client = genai.Client(api_key=API_KEY)
    models = client.models.list()
    gemma_models = [m.name for m in models if "gemma" in m.name.lower()]
    if gemma_models:
        for m in gemma_models:
            print(f"  {m}")
    else:
        print("  No Gemma models found — your project may not have access.")
        print("  Visit: https://aistudio.google.com to enable model access.")
except Exception as e:
    print(f"  Could not list models: {e}")

print("\nDone.")