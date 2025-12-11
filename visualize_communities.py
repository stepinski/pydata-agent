import json
import networkx as nx
from networkx.algorithms import community
from pyvis.network import Network
from pathlib import Path
from collections import Counter

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_FILE = Path("./output/knowledge_graph_enhanced.json")
OUTPUT_FILE = "community_graph.html"

# distinct colors for up to 10 communities
COMMUNITY_COLORS = [
    "#FF5733", "#33FF57", "#3357FF", "#F033FF", "#FF33A8", 
    "#33FFF5", "#F5FF33", "#FF8C33", "#8C33FF", "#33FF8C"
]

def get_community_tags(G_full, community_nodes):
    """
    Analyzes a community to find what defines it.
    Returns: Main Topic (str), Top 3 Keywords (str)
    """
    topics = []
    keywords = []
    
    for node in community_nodes:
        # Look at neighbors in the FULL graph (including keyword nodes)
        for neighbor in G_full.neighbors(node):
            node_type = G_full.nodes[neighbor].get('type')
            if node_type == 'Topic':
                topics.append(neighbor)
            elif node_type == 'Keyword':
                keywords.append(neighbor)

    # Find most common
    main_topic = Counter(topics).most_common(1)
    main_topic = main_topic[0][0] if main_topic else "Mixed Topics"
    
    top_kws = [k for k, v in Counter(keywords).most_common(4)]
    kw_str = ", ".join(top_kws)
    
    return main_topic, kw_str

def visualize_communities():
    print(f"📂 Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print("❌ File not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    G_full = nx.node_link_graph(data)
    
    # 1. Detect Communities (using the full rich graph)
    # This groups notes that share many keywords/topics
    print("🕵️  Detecting Communities...")
    communities = list(community.greedy_modularity_communities(G_full))
    
    # 2. Build the Simplified "Community View" Graph
    net = Network(height="90vh", width="100%", bgcolor="#222222", font_color="white", select_menu=True)
    
    # We will only add NOTE nodes to this visual
    processed_notes = set()
    
    print(f"\n📊 Community Analysis:")
    
    for i, comm in enumerate(communities):
        # Filter for only Note nodes in this community
        comm_notes = [n for n in comm if G_full.nodes[n].get('type') == 'Note']
        if not comm_notes: 
            continue
            
        # Get the "Identity" of this community
        topic_label, keywords_label = get_community_tags(G_full, comm_notes)
        color = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
        
        print(f"   🔹 Community {i+1} ({len(comm_notes)} files)")
        print(f"      Topic: {topic_label}")
        print(f"      Keywords: {keywords_label}")
        print(f"      Files: {', '.join(comm_notes)}\n")

        # Add Nodes to PyVis
        for note in comm_notes:
            processed_notes.add(note)
            
            # Tooltip explains WHY it is in this cluster
            tooltip = (
                f"📄 File: {note}\n"
                f"🏷️ Community #{i+1}\n"
                f"📌 Domain: {topic_label}\n"
                f"🔑 Key Themes: {keywords_label}"
            )
            
            net.add_node(
                note, 
                label=note, 
                title=tooltip, 
                color=color, 
                size=20,
                shape="dot",
                group=f"Community {i+1}"
            )

    # 3. Add Edges (Only Direct Note-to-Note connections)
    # We ignore edges to keywords. We only show if Note A is directly related to Note B.
    edge_count = 0
    for u, v, data in G_full.edges(data=True):
        if u in processed_notes and v in processed_notes:
            # This is a direct connection between notes
            relation = data.get('relation', 'related')
            weight = data.get('weight', 1)
            
            # Dynamic coloring based on relation
            color = "#555555" # Default grey
            width = 1
            dashed = False
            
            if relation == "shares_keywords":
                color = "#F1C40F" # Gold
                width = 2
            elif relation == "content_similar":
                color = "#3498DB" # Blue
                width = 3
            elif relation == "semantic_keywords":
                color = "#9B59B6" # Purple
                dashed = True
            
            net.add_edge(u, v, title=f"{relation}", color=color, width=width, dashes=dashed)
            edge_count += 1

    print(f"🔗 Drawn {edge_count} direct connections between notes.")

    # 4. Physics for Clustering
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "springLength": 100,
          "springConstant": 0.05,
          "damping": 0.4,
          "avoidOverlap": 1
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    net.write_html(OUTPUT_FILE)
    abs_path = Path(OUTPUT_FILE).resolve()
    print(f"✅ Clean visualization saved to: file://{abs_path}")

if __name__ == "__main__":
    visualize_communities()
