import os

# Force offline mode for all tests — models must be cached before running tests.
# This prevents HuggingFace from attempting network calls on every model load.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
