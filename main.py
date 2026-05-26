import os

# Must be set BEFORE any HuggingFace imports.
# Models are loaded from local cache — run `python download_models.py` once on first setup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import uvicorn
from docpipeline.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, reload=False)
