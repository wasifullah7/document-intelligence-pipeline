"""
First-time setup: download all required AI models to local cache.
Run this ONCE while connected to the internet.
After this, the system runs fully offline.
"""
import os

# Ensure we are NOT in offline mode during download
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

print("Downloading BAAI/bge-small-en-v1.5 (semantic search model, ~133 MB)...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5")
print("  Done.")

print("Downloading MoritzLaurer/deberta-v3-xsmall-zeroshot (classifier, ~142 MB)...")
from transformers import pipeline
pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    device=-1,
)
print("  Done.")

print("Downloading spaCy en_core_web_sm...")
import subprocess, sys
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
print("  Done.")

print("\nAll models cached. The system will now run fully offline.")
print("Start the API with:  python main.py")
