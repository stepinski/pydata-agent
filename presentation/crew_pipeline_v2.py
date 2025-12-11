# important note - remember to document!!!! i used patch to 
import warnings
import os

# Suppress warnings first, before importing anything else
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# Standard library imports
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict

# Third-party imports
from pdf2image import convert_from_path
from openai import OpenAI
from langchain_core.documents import Document
from PIL import Image
import torch
import cv2
import numpy as np

# ML/AI imports
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft

# CrewAI imports
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

# Config imports
from config import (
    DATA_DIR, 
    OUTPUT_DIR, 
    VISION_LLM_MODEL, 
    TEXT_LLM_MODEL, 
    VISION_LLM_BASE_URL, 
    TEXT_LLM_BASE_URL, 
    MLC_LLM_API_KEY
)

import json
from typing import Dict

# ==============================================================================
# STEP 1: OCR CORRECTION AGENT (NEW)
# ==============================================================================

class OCRCorrectionTool(BaseTool):
    name: str = "OCR Error Corrector"
    description: str = "Corrects OCR errors using context clues, phonetic similarity, and domain knowledge."

    def _run(self, raw_ocr_text: str) -> str:
        """
        Uses the LLM to fix OCR errors intelligently.
        Prompt gives the LLM context about common OCR mistakes.
        """
        correction_prompt = f"""You are an OCR error correction specialist. Your job is to fix common OCR mistakes in handwritten note transcriptions.

Common OCR error patterns:
- Similar looking letters: l/1, O/0, S/5, rn/m, a/d, n/u
- Phonetic mistakes: "tobacco" for "blood", "neoucation" for "medication"
- Missing punctuation and structure
- Context-based fixes: if it says "Insurance cord" it should be "Insurance card"

Raw OCR text:
{raw_ocr_text}

Instructions:
1. Identify likely OCR errors using context clues and phonetic similarity
2. Correct them to make semantic sense
3. Preserve the original structure and formatting
4. If uncertain, keep the original text
5. Output ONLY the corrected text, no explanations

Corrected text:"""
        
        try:
            response = text_llm.call({"messages": [{"role": "user", "content": correction_prompt}]})
            corrected_text = response.content if hasattr(response, 'content') else str(response)
            return corrected_text.strip()
        except Exception as e:
            print(f"  ⚠️  Correction failed: {e}")
            return raw_ocr_text


# Also suppress HuggingFace logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
# ==============================================================================
# STEP 1: INITIALIZE MODELS ONCE (Outside the Tool)
# ==============================================================================
# Use the 'large' model for maximum accuracy on handwriting
TROCR_MODEL_NAME = "microsoft/trocr-large-handwritten" 
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

try:
    # 1. Initialize TrOCR (Recognition Model)
    trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME).to(DEVICE)
    
    # 2. Initialize CRAFT (Detection Model)
    # craft_text_detector handles text line detection
    craft_detector = Craft(output_dir=None, crop_type="box", cuda=(DEVICE == "cuda"))
    print(f"✅ TrOCR and CRAFT models loaded on device: {DEVICE}")

except Exception as e:
    print(f"❌ Error initializing TrOCR/CRAFT models. Check PyTorch/Transformers installation: {e}")
    # Set to None for safe failure within the tool
    trocr_processor, trocr_model, craft_detector = None, None, None

# ==============================================================================
# STEP 1: SETUP LLMs
# ==============================================================================

# FIX: Prepend 'openai/' so LiteLLM knows to use the OpenAI protocol
text_llm = LLM(
    model=f"openai/{TEXT_LLM_MODEL}", 
    base_url=TEXT_LLM_BASE_URL,
    api_key=MLC_LLM_API_KEY,
    temperature=0.7
)

# Vision Client (Direct OpenAI usage, no LiteLLM wrapper needed here)
vision_client = OpenAI(
    base_url=VISION_LLM_BASE_URL,
    api_key=MLC_LLM_API_KEY
)

# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
#
# # Load the OCR model once
# print("🔧 Loading DocTR model...")
# doctr_model = ocr_predictor(
#     det_arch="db_resnet50",
#     reco_arch="crnn_vgg16_bn",
#     pretrained=True
# )
# print("✅ DocTR initialized")

# ==============================================================================
# STEP 2: CUSTOM TROCR TOOL
# ==============================================================================
class LocalVisionOCRTool(BaseTool):
    name: str = "TrOCR Handwritten Text Extractor"
    description: str = "Uses the high-accuracy TrOCR + CRAFT pipeline to extract and transcribe text line-by-line from a PDF, optimized for messy handwriting."
    
    def _enhance_image(self, pil_image):
        """Enhance image quality for better OCR results."""
        from PIL import ImageEnhance
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # Slightly increase brightness for faded text
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return pil_image
    
    def _run(self, pdf_path: str) -> str:
        if trocr_processor is None or trocr_model is None or craft_detector is None:
            return "Error: TrOCR/CRAFT models failed to initialize."
             
        if not os.path.exists(pdf_path):
            return "Error: File not found."
        
        print(f"    ... ✍️  Detecting and Transcribing PDF with TrOCR/CRAFT: {Path(pdf_path).name}")
        
        # --- PDF to Image Conversion ---
        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
            if not images:
                return "Error: Could not convert PDF to image."
            
            pil_image = images[0].convert("RGB")
            
            # Pre-process image for better OCR: increase contrast and sharpness
            pil_image = self._enhance_image(pil_image)
            
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return f"Error during image conversion: {e}"
        
        # --- Text Detection (CRAFT) ---
        try:
            result = craft_detector.detect_text(cv_image)
            boxes = result["boxes"]
            
            # If no boxes detected, return empty string
            if boxes is None or len(boxes) == 0:
                return "No text regions detected."
                
        except Exception as e:
            return f"Error during CRAFT text detection: {e}"
        
        # --- Text Recognition (TrOCR) ---
        extracted_text = []
        try:
            with torch.no_grad():
                for box in boxes:
                    try:
                        # box format is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        x_min, y_min = int(min(p[0] for p in box)), int(min(p[1] for p in box))
                        x_max, y_max = int(max(p[0] for p in box)), int(max(p[1] for p in box))
                        
                        # Add small padding to avoid edge crops
                        x_min, y_min = max(0, x_min - 5), max(0, y_min - 5)
                        x_max, y_max = min(pil_image.width, x_max + 5), min(pil_image.height, y_max + 5)
                        
                        # Crop and validate
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        
                        crop_line = pil_image.crop((x_min, y_min, x_max, y_max))
                        
                        # Process with TrOCR with better generation params
                        pixel_values = trocr_processor(crop_line, return_tensors="pt").pixel_values.to(DEVICE)
                        generated_ids = trocr_model.generate(
                            pixel_values,
                            max_new_tokens=128,
                            num_beams=4,  # Beam search for better accuracy
                            early_stopping=True
                        )
                        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Clean up text
                        text = text.strip()
                        if text and len(text) > 1:  # Filter out single character noise
                            extracted_text.append(text)
                    except Exception as box_error:
                        print(f"  ⚠️  Skipping problematic box: {box_error}")
                        continue
        except Exception as e:
            return f"Error during TrOCR processing: {e}"
        
        # --- Final Output ---
        final_transcription = "\n".join(extracted_text)
        
        # Clean up VRAM (but don't unload CRAFT — just empty cache)
        if DEVICE != 'cpu':
            torch.cuda.empty_cache()
            
        if not final_transcription.strip():
            return "No text could be reliably transcribed by TrOCR/CRAFT."
        
        return final_transcription

# ==============================================================================
# STEP 3: AGENTS
# ==============================================================================

ocr_agent = Agent(
    role="Digitization Specialist",
    goal="Convert physical handwritten notes into digital text.",
    backstory="You are an expert at deciphering difficult handwriting. You use vision tools to read documents.",
    tools=[LocalVisionOCRTool()],
    verbose=True,
    llm=text_llm 
)

ocr_correction_agent = Agent(
    role="OCR Error Corrector",
    goal="Fix OCR errors using context, phonetics, and domain knowledge.",
    backstory="You are an expert at correcting OCR mistakes in handwritten notes. You use semantic context and phonetic similarity to fix errors.",
    tools=[OCRCorrectionTool()],
    verbose=True,
    llm=text_llm
)

classification_agent = Agent(
    role="Organizer",
    goal="Categorize notes into: Meeting, Idea, Todo, or Reference.",
    backstory="You are a highly organized filing assistant. You categorize everything strictly.",
    verbose=True,
    llm=text_llm
)

knowledge_agent = Agent(
    role="Knowledge Extractor",
    goal="Extract entities, keywords, and summary from text.",
    backstory="You read notes and extract structured metadata for a database.",
    verbose=True,
    llm=text_llm
)

# ==============================================================================
# STEP 4: PIPELINE
# ==============================================================================

def process_single_note(pdf_path: str) -> Dict:
    print(f"\n{'='*60}\nProcessing: {Path(pdf_path).name}\n{'='*60}")
    
    # --- Task 1: OCR ---
    task_ocr = Task(
        description=f"Use the Vision Tool to read the file at: '{pdf_path}'. Return the raw text.",
        expected_output="The raw extracted text from the image.",
        agent=ocr_agent
    )

    task_correct = Task(
        description="Take the raw OCR text and correct common OCR errors using context clues, phonetic similarity, and domain knowledge. Fix mistakes like 'tobacco' → 'blood', 'Insurance cord' → 'Insurance card', etc.",
        expected_output="The corrected and cleaned text with OCR errors fixed.",
        agent=ocr_correction_agent,
        context=[task_ocr]
    )

    # --- Task 3: Classify ---
    task_classify = Task(
        description="Analyze the extracted text. Classify it (Meeting/Idea/Todo/Reference) with a confidence score.",
        expected_output="JSON string: {note_type, confidence, reasoning}",
        agent=classification_agent,
        context=[task_ocr]
    )

    # --- Task 4: Extract ---
    task_extract = Task(
        description="Extract the 'main_topic' (one sentence) and a list of 'keywords' from the text.",
        expected_output="JSON string: {main_topic, keywords}",
        agent=knowledge_agent,
        context=[task_ocr]
    )

    crew = Crew(
        agents=[ocr_agent, ocr_correction_agent, classification_agent, knowledge_agent],
        tasks=[task_ocr, task_correct, task_classify, task_extract],
        verbose=True
    )

    result = crew.kickoff()
    
    # Attempt to parse the final JSON output from the last task
    try:
        raw_output = str(task_extract.output).replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(raw_output)
    except:
        parsed_data = {"raw_output": str(result)}

    return {
        "file": Path(pdf_path).name,
        "data": parsed_data
    }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDFs found in {DATA_DIR}")
        dummy_path = DATA_DIR / "test_note.pdf"
        with open(dummy_path, "w") as f: f.write("%PDF-1.4...") 
        print(f"Created dummy file: {dummy_path}")
        pdf_files = [dummy_path]

    results = []
    for pdf in pdf_files:
        res = process_single_note(str(pdf))
        results.append(res)

    with open(OUTPUT_DIR / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Processed {len(results)} files. Check {OUTPUT_DIR}/final_results.json")
    # After processing all PDFs:
    print("Cleaning up models...")
    craft_detector.unload_craftnet_model()
    if DEVICE != 'cpu':
        torch.cuda.empty_cache()
    print("✅ Done!")
