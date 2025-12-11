import json
import networkx as nx
from pyvis.network import Network
from pathlib import Path

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_FILE = Path("./output/knowledge_graph_enhanced.json")
OUTPUT_FILE = "relation_viz.html"

# Target Nodes to Focus On
TARGET_A = "note_10.pdf"
TARGET_B = "note_03.pdf"

# Presentation Palette (High Contrast)
COLOR_NOTE_A = "#7aa2f7"  # Tokyo Night Blue
COLOR_NOTE_B = "#bb9af7"  # Tokyo Night Purple
COLOR_BRIDGE = "#e0af68"  # Tokyo Night Orange (for shared keywords)
COLOR_BG = "#1a1b26"      # Tokyo Night Dark Background
COLOR_TEXT = "#c0caf5"    # Soft White

def visualize_specific_relation():
    print(f"📂 Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    G_full = nx.node_link_graph(data)
    
    if TARGET_A not in G_full or TARGET_B not in G_full:
        print(f"❌ Targets {TARGET_A} or {TARGET_B} not found in graph!")
        return

    # --- 1. FILTER GRAPH (Create a Subgraph) ---
    # We want ONLY the two notes and the nodes that connect them.
    
    # Find shared neighbors (Keywords/Topics linking them)
    neighbors_a = set(G_full.neighbors(TARGET_A))
    neighbors_b = set(G_full.neighbors(TARGET_B))
    shared_nodes = neighbors_a.intersection(neighbors_b)
    
    # Also check if there is a direct edge
    has_direct = G_full.has_edge(TARGET_A, TARGET_B)
    
    # Collect all nodes for our focused graph
    focused_nodes = {TARGET_A, TARGET_B}
    focused_nodes.update(shared_nodes)
    
    # Create the subgraph
    G_viz = G_full.subgraph(focused_nodes).copy()
    
    print(f"🔍 Found {len(shared_nodes)} shared connections + Direct Link: {has_direct}")

    # --- 2. STYLE THE NODES ---
    for node in G_viz.nodes():
        attrs = G_viz.nodes[node]
        attrs["font"] = {"size": 20, "color": COLOR_TEXT, "face": "arial"}
        attrs["shape"] = "dot"
        
        if node == TARGET_A:
            attrs["color"] = COLOR_NOTE_A
            attrs["size"] = 40
            attrs["label"] = "Note 10\n(Strategy)"
            
        elif node == TARGET_B:
            attrs["color"] = COLOR_NOTE_B
            attrs["size"] = 40
            attrs["label"] = "Note 03\n(Tech Setup)"
            
        else:
            # These are the shared keywords/bridges
            attrs["color"] = COLOR_BRIDGE
            attrs["size"] = 25
            attrs["label"] = node # Actual keyword text

    # --- 3. STYLE THE EDGES ---
    for u, v, attrs in G_viz.edges(data=True):
        attrs["width"] = 3
        attrs["color"] = "#565f89" # Subtle blue-grey connection
        
        # Highlight direct connection if it exists
        if (u == TARGET_A and v == TARGET_B) or (u == TARGET_B and v == TARGET_A):
            attrs["width"] = 6
            attrs["color"] = "#9ece6a" # Bright Green for the direct semantic link
            attrs["dashes"] = True
            attrs["label"] = "Semantic Match"
            attrs["font"] = {"align": "middle", "strokeWidth": 0, "color": "#9ece6a"}

    # --- 4. BUILD PYVIS ---
    net = Network(height="800px", width="100%", bgcolor=COLOR_BG, font_color=COLOR_TEXT)
    net.from_nx(G_viz)

    # Physics settings to make it look like a nice "dumbbell" or "bowtie"
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "springLength": 200,
          "springConstant": 0.05
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    net.write_html(OUTPUT_FILE)
    abs_path = Path(OUTPUT_FILE).resolve()
    print(f"✅ Focused visualization saved to: file://{abs_path}")

if __name__ == "__main__":
    visualize_specific_relation()
