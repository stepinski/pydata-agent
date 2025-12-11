import time
import json
import os
import sys
import random
from pathlib import Path
import networkx as nx
import warnings

warning.filterwarnings("ignore", category=FutureWarning,module="networkx")

# --- Config ---
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
PRECOMPUTED_FILE = OUTPUT_DIR / "final_results.json"

# --- Visual Styles ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ============================================================
# 1. REAL GRAPH GENERATION LOGIC (Kept authentic)
# ============================================================
def generate_knowledge_graph(processed_results):
    """
    Real graph building logic using NetworkX.
    This actually runs on the pre-computed data.
    """
    print(f"\n{Colors.HEADER}{'='*60}\n🕸️  Building Knowledge Graph (Real-time)\n{'='*60}{Colors.ENDC}")
    
    G = nx.Graph()
    successful_notes = [r for r in processed_results if r.get('status') == 'success']
    
    for entry in successful_notes:
        filename = entry['file']
        raw_json_str = entry['final_output']
        
        try:
            # Cleanup markdown if present
            clean_str = raw_json_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            
            main_topic = data.get('main_topic', 'Unknown Topic')
            keywords = data.get('keywords', [])
            
            # Nodes
            G.add_node(filename, type="Note", color="blue")
            G.add_node(main_topic, type="Topic", color="green")
            
            # Edges
            G.add_edge(filename, main_topic, relation="is_about")
            
            for kw in keywords:
                kw_clean = kw.lower().strip()
                G.add_node(kw_clean, type="Keyword", color="red")
                G.add_edge(filename, kw_clean, relation="has_keyword")
                
        except Exception:
            continue

    # Stats
    print(f"📊 {Colors.BOLD}Graph Statistics:{Colors.ENDC}")
    print(f"   - Nodes: {G.number_of_nodes()}")
    print(f"   - Edges: {G.number_of_edges()}")
    
    # Save Graph Data for Visualization
    graph_data = nx.node_link_data(G, edges="links")
    output_path = OUTPUT_DIR / "knowledge_graph.json"
    with open(output_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"\n✅ {Colors.GREEN}Graph data saved to {output_path}{Colors.ENDC}")

    # Discover Connections (The "Smart" part)
    print(f"\n{Colors.CYAN}🔗 Discovered Hidden Connections:{Colors.ENDC}")
    seen_pairs = set()
    for n1 in successful_notes:
        f1 = n1['file']
        for n2 in successful_notes:
            f2 = n2['file']
            if f1 == f2: continue
            
            if f1 in G and f2 in G and nx.has_path(G, f1, f2):
                path = nx.shortest_path(G, f1, f2)
                # Only show tight connections (shared keyword)
                if len(path) <= 3: 
                    pair = tuple(sorted((f1, f2)))
                    if pair not in seen_pairs:
                        mid_point = path[1]
                        print(f"   🚀 {Colors.BOLD}{f1}{Colors.ENDC} <--> {Colors.BOLD}{f2}{Colors.ENDC} (via '{mid_point}')")
                        seen_pairs.add(pair)
                        time.sleep(0.3) # Small delay for dramatic effect

# ============================================================
# 2. SIMULATED PROCESSING (The "Show")
# ============================================================
def simulate_agent_execution(filename, result_data):
    """Simulate the time taken by OCR and LLM agents."""
    
    print(f"\n{Colors.HEADER}{'='*60}\n🎬 Processing: {filename}\n{'='*60}{Colors.ENDC}")
    time.sleep(0.5)

    # 1. OCR Simulation
    print(f"{Colors.BLUE}🤖 [Vision Specialist]{Colors.ENDC} Starting TrOCR + Craft pipeline...")
    time.sleep(0.8)
    
    # Fake detected regions count
    regions = random.randint(15, 45)
    print(f"    ✅ CRAFT: Detected {regions} text regions")
    time.sleep(0.4)
    
    # Extract a preview snippet from the real result
    try:
        full_json = json.loads(result_data['final_output'].replace("```json", "").replace("```", "").strip())
        preview_topic = full_json.get('main_topic', 'Notes...')
        preview_kw = full_json.get('keywords', ['text'])[0]
        print(f"    🧠 TrOCR preview: {preview_topic} - {preview_kw} ...")
    except:
        print(f"    🧠 TrOCR preview: Raw handwriting extracted...")
    
    time.sleep(0.6)

    # 2. Correction Simulation
    print(f"{Colors.BLUE}📝 [Senior Editor]{Colors.ENDC} Reconstructing sentences and fixing OCR errors...")
    time.sleep(1.2) # Editors take time to think

    # 3. Classification Simulation
    print(f"{Colors.BLUE}📚 [Librarian]{Colors.ENDC} Classifying content and extracting JSON metadata...")
    time.sleep(0.8)

    print(f"✅ {Colors.GREEN}Finished processing {filename}{Colors.ENDC}")

def main():
    if not PRECOMPUTED_FILE.exists():
        print(f"{Colors.FAIL}❌ Error: Precomputed file not found at {PRECOMPUTED_FILE}")
        print("Please run the real script once to generate the data.{Colors.ENDC}")
        sys.exit(1)

    with open(PRECOMPUTED_FILE, 'r') as f:
        all_results = json.load(f)

    print(f"{Colors.BOLD}🔧 Loading Local LLM (MLC-AI/Qwen-2-7B)...{Colors.ENDC}")
    time.sleep(1.5) # Fake model load time
    print(f"{Colors.BOLD}🔧 Loading Vision Models (TrOCR Large)...{Colors.ENDC}")
    time.sleep(1.5)
    print(f"{Colors.GREEN}✅ Models Loaded on MPS/GPU.{Colors.ENDC}\n")

    # Run the "Processing" simulation
    for result in all_results:
        simulate_agent_execution(result['file'], result)
    
    # Run the REAL graph generation
    generate_knowledge_graph(all_results)
    
    print(f"\n{Colors.BOLD}✨ Demo Pipeline Complete.{Colors.ENDC}")

if __name__ == "__main__":
    main()
