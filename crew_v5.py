import warnings
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Optional, Type, List

# --- CrewAI Network Bypass ---
os.environ.setdefault("CREWAI_TELEMETRY", "False")
os.environ.setdefault("CREWAI_TRACING", "False")

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# --- Graph Theory Import ---
try:
    import networkx as nx
except ImportError:
    print("❌ Critical Error: networkx not found. Run 'pip install networkx'")
    sys.exit(1)

from PIL import Image, ImageEnhance, ImageOps
import torch
import cv2
import numpy as np
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft, craft_utils  # Added craft_utils for patching

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel

# ============================================================
# 🔧 CRITICAL FIX: MONKEY PATCH FOR NUMPY COMPATIBILITY
# ============================================================
# This fixes the "ValueError: setting an array element with a sequence"
# by sanitizing polygons before NumPy tries to process them.
def fixed_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        clean_polys = []
        for p in polys:
            p = np.array(p)
            # Only keep valid 4-point boxes
            if p.shape == (4, 2):
                clean_polys.append(p)
        
        # If we have valid boxes, use them; otherwise empty array
        polys = np.array(clean_polys) if len(clean_polys) > 0 else np.array([])

        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

# Apply the patch to the library
craft_utils.adjustResultCoordinates = fixed_adjustResultCoordinates
# ============================================================

# --- Import project config ---
try:
    from config import DATA_DIR, OUTPUT_DIR, MLC_LLM_BASE_URL, MLC_LLM_API_KEY
except ImportError:
    # Fallback config for testing if config.py is missing
    DATA_DIR = Path("./data")
    OUTPUT_DIR = Path("./output")
    MLC_LLM_BASE_URL = "http://localhost:8080/v1"
    MLC_LLM_API_KEY = "dummy"

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
#        ENHANCED KNOWLEDGE GRAPH WITH SEMANTIC LINKING
# ============================================================

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
MAX_CONTEXT_TOKENS = 128000

# For semantic similarity (install: pip install sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    print("⚠️  sentence-transformers not found. Install with: pip install sentence-transformers")
    print("    Falling back to keyword-only matching (limited relations)")

# ============================================================
#              SEMANTIC SIMILARITY FUNCTIONS
# ============================================================

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two text snippets (0-1)."""
    if not HAS_SEMANTIC:
        return 0.0
    
    emb1 = SEMANTIC_MODEL.encode(text1, convert_to_tensor=True)
    emb2 = SEMANTIC_MODEL.encode(text2, convert_to_tensor=True)
    
    # Cosine similarity
    sim = float((emb1 @ emb2) / (emb1.norm() * emb2.norm()))
    return max(0.0, sim)  # Clamp to [0, 1]

def find_semantic_matches(
    query: str, 
    candidates: List[str], 
    threshold: float = 0.6
) -> List[Tuple[str, float]]:
    """Find semantically similar strings above threshold."""
    matches = []
    for candidate in candidates:
        sim = calculate_semantic_similarity(query, candidate)
        if sim >= threshold:
            matches.append((candidate, sim))
    return sorted(matches, key=lambda x: x[1], reverse=True)

# ============================================================
#            ENHANCED GRAPH BUILDER WITH MULTIPLE STRATEGIES
# ============================================================

def generate_knowledge_graph_enhanced(processed_results: List[Dict], OUTPUT_DIR: Path):
    """
    Enhanced graph building with:
    1. Direct keyword matching (original)
    2. Semantic keyword similarity
    3. Topic-based clustering
    4. Content-based similarity between notes
    """
    print(f"\n{'='*60}\n🕸️  Building Enhanced Knowledge Graph\n{'='*60}")
    
    G = nx.Graph()
    successful_notes = [r for r in processed_results if r['status'] == 'success']
    
    # Parse all note data
    note_data = {}
    for entry in successful_notes:
        filename = entry['file']
        raw_json_str = entry['final_output']
        
        try:
            clean_str = raw_json_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            note_data[filename] = {
                'topic': data.get('main_topic', 'Unknown'),
                'keywords': [kw.lower().strip() for kw in data.get('keywords', [])],
                'raw_text': entry.get('raw_text', '')  # If available
            }
        except json.JSONDecodeError:
            print(f"⚠️  Skipping {filename}")
            continue
    
    # --- PHASE 1: Add nodes ---
    for filename, data in note_data.items():
        G.add_node(filename, type="Note", color="blue")
        
        topic = data['topic']
        G.add_node(topic, type="Topic", color="green")
        G.add_edge(filename, topic, relation="is_about", weight=1.0)
        
        for kw in data['keywords']:
            G.add_node(kw, type="Keyword", color="red")
            G.add_edge(filename, kw, relation="has_keyword", weight=1.0)
    
    # --- PHASE 2: Direct keyword matching (original approach) ---
    print("\n📍 Strategy 1: Direct Keyword Matching")
    direct_matches = 0
    seen_pairs = set()
    
    for f1 in note_data:
        for f2 in note_data:
            if f1 >= f2: continue  # Avoid duplicates
            
            kw1 = set(note_data[f1]['keywords'])
            kw2 = set(note_data[f2]['keywords'])
            shared = kw1 & kw2
            
            if shared:
                pair = tuple(sorted((f1, f2)))
                if pair not in seen_pairs:
                    print(f"    ✓ {f1} <--> {f2} (shared: {', '.join(shared)})")
                    G.add_edge(f1, f2, relation="shares_keywords", shared_kw=list(shared), weight=2.0)
                    seen_pairs.add(pair)
                    direct_matches += 1
    
    print(f"    Found: {direct_matches} direct connections")
    
    # --- PHASE 3: Semantic keyword matching (NEW) ---
    if HAS_SEMANTIC:
        print("\n🧠 Strategy 2: Semantic Keyword Similarity")
        semantic_matches = 0
        
        for f1 in note_data:
            kw1 = note_data[f1]['keywords']
            for f2 in note_data:
                if f1 >= f2: continue
                if (f1, f2) in seen_pairs or (f2, f1) in seen_pairs: continue
                
                kw2 = note_data[f2]['keywords']
                
                # Find semantic matches between keyword lists
                matches_found = []
                for k1 in kw1:
                    similar = find_semantic_matches(k1, kw2, threshold=0.65)
                    if similar:
                        matches_found.append((k1, similar[0][0], similar[0][1]))
                
                if matches_found:
                    pair = tuple(sorted((f1, f2)))
                    match_str = ", ".join([f"'{m[0]}' ≈ '{m[1]}'" for m in matches_found])
                    print(f"    ✓ {f1} <--> {f2} ({match_str})")
                    G.add_edge(f1, f2, relation="semantic_keywords", matches=match_str, weight=1.5)
                    seen_pairs.add(pair)
                    semantic_matches += 1
        
        print(f"    Found: {semantic_matches} semantic connections")
        
        # --- PHASE 4: Topic-based clustering (NEW) ---
        print("\n🎯 Strategy 3: Topic Clustering")
        topic_matches = 0
        topic_groups = {}
        
        for f1, data1 in note_data.items():
            topic1 = data1['topic']
            if topic1 not in topic_groups:
                topic_groups[topic1] = []
            topic_groups[topic1].append(f1)
        
        # Connect notes with similar topics
        for topic, files in topic_groups.items():
            if len(files) > 1:
                for i, f1 in enumerate(files):
                    for f2 in files[i+1:]:
                        pair = tuple(sorted((f1, f2)))
                        if pair not in seen_pairs:
                            print(f"    ✓ {f1} <--> {f2} (both about '{topic}')")
                            G.add_edge(f1, f2, relation="same_topic", topic=topic, weight=1.2)
                            seen_pairs.add(pair)
                            topic_matches += 1
        
        print(f"    Found: {topic_matches} topic-based connections")
        
        # --- PHASE 5: Content similarity (ADVANCED, optional) ---
        print("\n📄 Strategy 4: Content Similarity (Full Text)")
        content_matches = 0
        
        # Only if raw_text is available
        files_with_text = {f: d for f, d in note_data.items() if d['raw_text']}
        
        if files_with_text:
            text_list = list(files_with_text.keys())
            for i, f1 in enumerate(text_list):
                for f2 in text_list[i+1:]:
                    pair = tuple(sorted((f1, f2)))
                    if pair in seen_pairs: continue
                    
                    sim = calculate_semantic_similarity(
                        note_data[f1]['raw_text'][:500],  # First 500 chars
                        note_data[f2]['raw_text'][:500]
                    )
                    
                    if sim >= 0.55:  # Fairly high threshold
                        print(f"    ✓ {f1} <--> {f2} (content similarity: {sim:.2f})")
                        G.add_edge(f1, f2, relation="content_similar", similarity=sim, weight=sim)
                        seen_pairs.add(pair)
                        content_matches += 1
            
            print(f"    Found: {content_matches} content-based connections")
    
    # --- STATISTICS ---
    print(f"\n📊 Final Graph Statistics:")
    print(f"    - Nodes: {G.number_of_nodes()}")
    print(f"    - Edges: {G.number_of_edges()}")
    print(f"    - Total Note-to-Note Relations: {sum(1 for u, v in G.edges() if 'Note' in str(G.nodes[u].get('type', '')) and 'Note' in str(G.nodes[v].get('type', '')))}")
    
    # Community detection
    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(G))
        print(f"    - Communities Found: {len(communities)}")
        for i, comm in enumerate(communities):
            notes_in_comm = [n for n in comm if G.nodes[n].get('type') == 'Note']
            if notes_in_comm:
                print(f"      Community {i+1}: {', '.join(notes_in_comm)}")
    except:
        pass
    
    # --- SAVE RESULTS ---
    graph_data = nx.node_link_data(G)
    with open(OUTPUT_DIR / "knowledge_graph_enhanced.json", "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"\n✅ Enhanced graph saved to {OUTPUT_DIR / 'knowledge_graph_enhanced.json'}")
    
    return G

# ============================================================
#                    IMPROVED HELPER FUNCTIONS
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
#                    TOOL DEFINITION
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
            
            # 2. PREPROCESSING PIPELINE
            # A: Clean image for DETECTION (standard contrast, no dilation)
            detect_enhancer = ImageEnhance.Contrast(pil_img)
            img_for_detect = detect_enhancer.enhance(1.5)
            cv_img_detect = cv2.cvtColor(np.array(img_for_detect), cv2.COLOR_RGB2BGR)

            # B: Thickened image for READING/TrOCR (high contrast + dilation)
            pil_img_enhanced = preprocess_image_for_trocr(pil_img)

            # 3. Detect (Using the CLEAN image to avoid blob errors)
            results = craft_detector.detect_text(cv_img_detect)
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

            # 5. Inference (Cropping from the ENHANCED image)
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

                    # Crop from the ENHANCED image for better character recognition
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
#                    SETUP & MAIN
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
        max_input_tokens=MAX_CONTEXT_TOKENS,
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
        description="""Extract the main topic and keywords from the corrected notes. 
        Keywords MUST be simple, single words or short phrases (e.g., 'Vector DB', 'Meeting Minutes', 'Python', 'Finance'). Do NOT use long sentences as keywords. Return ONLY valid JSON in the following format, with no other text:
        {
            "main_topic": "The general category or subject",
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }""",
        expected_output='JSON Object',
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
        return {"file": filename, "status": "success", "final_output": str(result), "raw_text": str(result)}
    except Exception as e:
        return {"file": filename, "status": "error", "error": str(e)}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_files = check_data_exists()
    if not pdf_files or not load_ocr_models(): return

    # 1. Processing Pipeline
    results = [process_single_note(str(pdf)) for pdf in pdf_files]
    
    # 2. Save Intermediate Results
    with open(OUTPUT_DIR / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 3. Generate Knowledge Graph (Next Pipeline Step)
    if any(r['status'] == 'success' for r in results):
        generate_knowledge_graph_enhanced(results, OUTPUT_DIR)
    
    print(f"\n✅ Done!")

if __name__ == "__main__":
    main()
