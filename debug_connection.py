import json
import networkx as nx
from pathlib import Path

INPUT_FILE = Path("./output/knowledge_graph_enhanced.json")

def debug_connection(file_a, file_b):
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    
    G = nx.node_link_graph(data)
    
    print(f"\n🕵️‍♂️ Inspecting connection: {file_a} <---> {file_b}")
    
    # 1. Check Direct Edge
    if G.has_edge(file_a, file_b):
        edge_data = G.get_edge_data(file_a, file_b)
        print(f"❌ DIRECT LINK FOUND!")
        print(f"   Reason: {edge_data.get('relation')}")
        print(f"   Details: {edge_data}")
    else:
        print("✅ No direct link (Good). Checking indirect paths...")

    # 2. Check Shared Neighbors (The likely culprit)
    common_neighbors = list(nx.common_neighbors(G, file_a, file_b))
    if common_neighbors:
        print(f"⚠️  They share these nodes:")
        for neighbor in common_neighbors:
            node_type = G.nodes[neighbor].get('type', 'Unknown')
            print(f"   - [{node_type}] {neighbor}")
    else:
        print("   No shared neighbors found.")

# Run the debug
debug_connection("note_06.pdf", "note_08.pdf")
