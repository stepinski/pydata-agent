from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MLC_DEVICE = "auto"

# MLC LLM models
VISION_LLM_MODEL = "HF://mlc-ai/Phi-3.5-vision-instruct-q0f16-MLC"
TEXT_LLM_MODEL   = "HF://mlc-ai/Qwen3-8B-q4f16_1-MLC"

# Servers
VISION_LLM_BASE_URL = "http://localhost:8081/v1"
TEXT_LLM_BASE_URL   = "http://localhost:8080/v1"
MLC_LLM_API_KEY     = "mlc-ai"
MLC_LLM_BASE_URL = "http://localhost:8080/v1"

