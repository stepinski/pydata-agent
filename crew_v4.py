import warnings
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Optional, Type

# --- CrewAI Network Bypass ---
os.environ.setdefault("CREWAI_TELEMETRY", "False")
os.environ.setdefault("CREWAI_TRACING", "False")

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from PIL import Image, ImageEnhance, ImageOps
import torch
import cv2
import numpy as np
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel

# --- Import project config ---
try:
    from config import DATA_DIR, OUTPUT_DIR, MLC_LLM_BASE_URL, MLC_LLM_API_KEY
except ImportError:
    print("❌ Critical Error: Could not import config.py")
    sys.exit(1)

# --- Device selection ---
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and hasattr(torch, "mps")
    else "cpu"
)

trocr_processor: Optional[TrOCRProcessor] = None
trocr_model: Optional[VisionEncoderDecoderModel] = None
craft_detector: Optional[Craft] = None

# ============================================================
#                   IMPROVED HELPER FUNCTIONS
# ============================================================

def preprocess_image_for_trocr(pil_img: Image.Image) -> Image.Image:
    """
    Advanced preprocessing:
    1. Grayscale & Contrast enhancement.
    2. Morphological dilation to 'thicken' thin handwriting strokes.
    """
    # 1. Convert to RGB and Contrast
    img = pil_img.convert("RGB")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Stronger contrast
    
    # 2. Sharpen
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    # 3. Morphological Dilation (Thickening) 
    # This helps TrOCR read faint or thin pen strokes by making them bolder.
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    kernel = np.ones((2, 2), np.uint8)
    # Dilate: Expands white regions (text) slightly
    cv_img = cv2.dilate(cv_img, kernel, iterations=1)
    
    # Back to PIL
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def merge_boxes(boxes: list) -> np.ndarray:
    """Merge multiple boxes into one encompassing box."""
    if not boxes:
        return np.array([])
    all_points = np.vstack(list(boxes))
    x_min = np.min(all_points[:, 0])
    x_max = np.max(all_points[:, 0])
    y_min = np.min(all_points[:, 1])
    y_max = np.max(all_points[:, 1])
    
    return np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])

def sort_regions(regions: list, line_tolerance: int = 25) -> list:
    """Sorts regions Top-to-Bottom, Left-to-Right."""
    if regions is None or len(regions) == 0:
        return []

    boxes = list(regions)
    # Sort by Y top-edge roughly
    boxes = sorted(boxes, key=lambda b: np.min(b[:, 1]))
    
    lines = []
    current_line = [boxes[0]]
    
    for i in range(1, len(boxes)):
        box = boxes[i]
        last_box = current_line[-1]
        
        y_center_last = np.mean(last_box[:, 1])
        y_center_curr = np.mean(box[:, 1])
        
        # If close vertically, consider same line
        if abs(y_center_curr - y_center_last) < line_tolerance:
            current_line.append(box)
        else:
            lines.extend(sorted(current_line, key=lambda b: np.min(b[:, 0])))
            current_line = [box]
            
    lines.extend(sorted(current_line, key=lambda b: np.min(b[:, 0])))
    return lines

def merge_nearby_regions(regions: list, x_threshold: int = 70, y_threshold: int = 30) -> list:
    """
    Aggressively merges horizontal text regions.
    Increased thresholds to better capture full sentences in one go.
    """
    if regions is None or len(regions) < 2:
        return regions if regions is not None else []

    sorted_regions = sort_regions(regions, line_tolerance=y_threshold)
    
    merged = []
    if not sorted_regions:
        return merged

    current_group = [sorted_regions[0]]
    
    for i in range(1, len(sorted_regions)):
        box = sorted_regions[i]
        last_box = current_group[-1]
        
        last_x_max = np.max(last_box[:, 0])
        last_y_mean = np.mean(last_box[:, 1])
        
        curr_x_min = np.min(box[:, 0])
        curr_y_mean = np.mean(box[:, 1])
        
        h_dist = curr_x_min - last_x_max
        v_dist = abs(curr_y_mean - last_y_mean)
        
        if v_dist < y_threshold and h_dist < x_threshold:
            current_group.append(box)
        else:
            merged.append(merge_boxes(current_group))
            current_group = [box]
            
    merged.append(merge_boxes(current_group))
    
    if len(merged) < len(regions):
        print(f"    🔗 Merged regions: {len(regions)} → {len(merged)}")
        
    return merged

def add_padding(pil_img: Image.Image, padding: int = 15) -> Image.Image:
    """Adds white padding. TrOCR needs context borders."""
    w, h = pil_img.size
    new_w = w + (padding * 2)
    new_h = h + (padding * 2)
    result = Image.new(pil_img.mode, (new_w, new_h), (255, 255, 255))
    result.paste(pil_img, (padding, padding))
    return result

# ============================================================
#                   TOOL DEFINITION
# ============================================================
class OCRToolInput(BaseModel):
    file_path: str

class LocalVisionOCRTool(BaseTool):
    name: str = "TrOCR File Reader"
    description: str = "Reads a handwritten PDF file. Returns raw text."
    args_schema: Type[BaseModel] = OCRToolInput

    def run(self, **kwargs):
        file_path = kwargs.get("file_path") or kwargs.get("file")
        return self._run(file_path)

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        try:
            # 1. PDF to Image
            images = convert_from_path(file_path, first_page=1, last_page=1, dpi=300)
            pil_img = images[0].convert("RGB")
            
            # 2. Advanced Preprocessing (Thickening + Contrast)
            pil_img_enhanced = preprocess_image_for_trocr(pil_img)
            cv_img = cv2.cvtColor(np.array(pil_img_enhanced), cv2.COLOR_RGB2BGR)

            # 3. Detect
            results = craft_detector.detect_text(cv_img)
            regions = []
            if isinstance(results, dict):
                if "boxes" in results: regions = results["boxes"]
                elif isinstance(results.get("text_boxes"), list):
                    regions = [b["box"] for b in results["text_boxes"] if "box" in b]
            regions = list(regions)

            print(f"    ✅ CRAFT: Detected {len(regions)} regions")

            # 4. Merge
            regions = merge_nearby_regions(regions, x_threshold=70, y_threshold=30)
            extracted_lines = []

            # 5. Inference
            with torch.no_grad():
                for box in regions:
                    pts = np.array(box)
                    x_min = max(0, int(pts[:, 0].min()))
                    y_min = max(0, int(pts[:, 1].min()))
                    x_max = min(pil_img_enhanced.width, int(pts[:, 0].max()))
                    y_max = min(pil_img_enhanced.height, int(pts[:, 1].max()))

                    # --- NOISE FILTER ---
                    w, h = x_max - x_min, y_max - y_min
                    # Skip if box is too small (removes dust/dots/stray marks)
                    if w < 25 or h < 25:
                        continue

                    crop = pil_img_enhanced.crop((x_min, y_min, x_max, y_max))
                    crop = add_padding(crop, padding=15)

                    pixel_values = trocr_processor(crop, return_tensors="pt").pixel_values.to(DEVICE)
                    generated_ids = trocr_model.generate(
                        pixel_values,
                        max_new_tokens=64,
                        num_beams=4,
                        early_stopping=True
                    )
                    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    
                    if text and len(text) > 2: # Filter out single char noise
                        extracted_lines.append(text)

            # Cleanup
            if DEVICE == "cuda": torch.cuda.empty_cache()
            
            result = "\n".join(extracted_lines)
            preview = result[:80].replace("\n", " ")
            print(f"    🧠 TrOCR preview: {preview}...")
            
            return result if result else "No text detected."

        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"OCR Pipeline Failed:\n{tb}")

# ============================================================
#                   SETUP & MAIN
# ============================================================
def load_ocr_models():
    global trocr_processor, trocr_model, craft_detector
    print(f"🔧 Loading OCR Models on {DEVICE}...")
    try:
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(DEVICE)
        craft_detector = Craft(output_dir=None, crop_type="box", cuda=(DEVICE == "cuda"))
        return True
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return False

def check_data_exists():
    if not DATA_DIR.exists(): return []
    return list(DATA_DIR.glob("*.pdf"))

def process_single_note(pdf_path: str) -> Dict:
    filename = Path(pdf_path).name
    print(f"\n{'='*60}\n🎬 Processing: {filename}\n{'='*60}")

    text_llm = LLM(
        model="openai/qwen",
        base_url=MLC_LLM_BASE_URL,
        api_key=MLC_LLM_API_KEY,
        temperature=0.3,
    )

    # Agents
    ocr_agent = Agent(
        role="Vision Specialist",
        goal="Extract raw text",
        backstory="OCR expert.",
        tools=[LocalVisionOCRTool()],
        verbose=True,
        llm=text_llm
    )

    classifier_agent = Agent(
        role="Librarian",
        goal="Classify content",
        backstory="Expert at categorizing documents.",
        verbose=True,
        llm=text_llm
    )

    corrector_agent = Agent(
        role="Senior Editor",
        goal="Reconstruct coherent notes from noisy OCR data",
        backstory="""You are a world-class editor. You receive noisy, broken text from an OCR machine.
        Your job is to deduce the original meaning based on context, domain knowledge, and semantic coherence.""",
        verbose=True,
        llm=text_llm
    )

    # Tasks
    task_ocr = Task(
        description=f"Extract text from '{pdf_path}'. Return ONLY the raw text.",
        expected_output="Raw OCR text.",
        agent=ocr_agent,
        input={"file_path": pdf_path}
    )

    task_context = Task(
        description="Analyze the raw text. Determine the 'Domain' (e.g., Medical, Tech, Finance, Philosophy).",
        expected_output="The Domain Name.",
        agent=classifier_agent,
        context=[task_ocr]
    )

    # --- GENERALIZED PROMPT (No hardcoded strings) ---
    task_correct = Task(
        description="""Review the Raw OCR Text and the Domain provided by the previous tasks.
        
        1. **Domain Alignment:** Use the identified Domain to interpret ambiguous words. (e.g., if Domain is Tech, 'pine' might be 'Pinecone', not a tree).
        2. **Noise Removal:** Remove random characters (like '0', 'r.', 'c.') that do not form words.
        3. **Coherence:** Reconstruct broken sentences. Merge lines that belong together.
        4. **Hallucination Check:** If a line makes absolutely no sense in the context of the Domain, discard it rather than inventing a meaning.
        
        Output the final notes in a clean, Markdown bullet-point format.""",
        expected_output="Clean, coherent markdown notes.",
        agent=corrector_agent,
        context=[task_ocr, task_context]
    )

    task_json = Task(
        description="Extract main topic and keywords.",
        expected_output='{"main_topic": "...", "keywords": [...]}',
        agent=classifier_agent,
        context=[task_correct]
    )

    crew = Crew(
        agents=[ocr_agent, classifier_agent, corrector_agent],
        tasks=[task_ocr, task_context, task_correct, task_json],
        verbose=True
    )

    try:
        result = crew.kickoff()
        return {"file": filename, "status": "success", "final_output": str(result)}
    except Exception as e:
        return {"file": filename, "status": "error", "error": str(e)}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_files = check_data_exists()
    if not pdf_files or not load_ocr_models(): return

    results = [process_single_note(str(pdf)) for pdf in pdf_files]
    
    with open(OUTPUT_DIR / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Done!")

if __name__ == "__main__":
    main()
