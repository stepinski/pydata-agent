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
OUTPUT_FILE = Path("knowledge_graph_communities_interactive.html")

COMMUNITY_COLORS = [
    "#FF5733", "#3357FF", "#3357FF", "#F033FF", "#FF33A8", 
    "#33FFF5", "#F5FF33", "#FF8C33", "#8C33FF", "#33FF8C"
]

# ==========================================
# 🛠️ JAVASCRIPT & CSS LOGIC
# ==========================================
# This script injects a custom "Floating Card" that appears on hover.
JS_INTERACTIVITY = """
<style>
    /* THE CUSTOM POPUP CARD */
    #custom-tooltip {
        position: absolute;
        display: none; /* Hidden by default */
        z-index: 9999;
        pointer-events: none; /* Let mouse pass through so it doesn't flicker */
        
        /* TOKYO NIGHT THEME */
        background-color: #1f2335; 
        border: 1px solid #7aa2f7;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        padding: 12px;
        min-width: 200px;
        max-width: 300px;
        
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        line-height: 1.5;
        color: #c0caf5;
    }
    
    #custom-tooltip strong {
        display: block;
        color: #fff;
        font-size: 15px;
        margin-bottom: 6px;
        border-bottom: 1px solid #3b4261;
        padding-bottom: 4px;
    }
    
    .tt-label { color: #7dcfff; font-weight: 600; font-size: 12px; }
    .tt-val { color: #fff; }
    
    /* RESET vis-tooltip styles (just in case they reappear) */
    div.vis-tooltip {
        visibility: hidden !important;
        display: none !important;
    }

    /* 1. RESET the default Vis.js tooltip container to be invisible/unobtrusive */
    div.vis-tooltip {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        z-index: 99999 !important;
        overflow: visible !important;
    }

    /* 2. STYLE our custom inner card */
    .vis-tooltip-body {
        background-color: #1f2335; /* Tokyo Night Dark Blue */
        border: 1px solid #7aa2f7; /* Bright Blue Border */
        border-radius: 8px;
        box-shadow: 0px 8px 15px rgba(0,0,0,0.5);
        padding: 12px 16px;
        min-width: 250px;
        max-width: 320px;
        color: #c0caf5; /* Soft White Text */
    }

    .tt-header {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 8px;
        border-bottom: 1px solid #3b4261;
        padding-bottom: 4px;
        display: block;
    }

    .tt-row {
        margin-bottom: 4px;
        font-size: 13px;
        line-height: 1.4;
    }

    .tt-label {
        color: #7dcfff; /* Cyan for labels */
        font-weight: 600;
        margin-right: 5px;
    }

    .tt-val {
        color: #fff;
    }
    
    .tt-val-dim {
        color: #9aa5ce; /* Dimmed text for keywords */
    }
</style>

<div id="custom-tooltip"></div>

<script type="text/javascript">
    var tooltipEl = document.getElementById('custom-tooltip');

    // --- 1. HOVER EVENT (Show Card) ---
    network.on("hoverNode", function (params) {
        var nodeId = params.node;
        var node = nodes.get(nodeId);
        
        if (node.custom_topic) {
            tooltipEl.innerHTML = `
                <strong>📄 ${node.label}</strong>
                <div><span class="tt-label">🏷️ GROUP:</span> <span class="tt-val">${node.custom_topic}</span></div>
                <div style="margin-top:2px;"><span class="tt-label">🔑 KEYS:</span> <span class="tt-val" style="color:#9aa5ce">${node.custom_keywords}</span></div>
            `;
            
            var nodePos = network.getPositions([nodeId])[nodeId];
            var screenPos = network.canvasToDOM(nodePos);
            
            tooltipEl.style.left = (screenPos.x + 20) + 'px';
            tooltipEl.style.top = (screenPos.y - 20) + 'px';
            tooltipEl.style.display = 'block';
        }
    });

    // --- 2. BLUR/DRAG/ZOOM EVENT (Hide Card) ---
    network.on("blurNode", function (params) { tooltipEl.style.display = 'none'; });
    network.on("dragging", function (params) { tooltipEl.style.display = 'none'; });
    network.on("zoom", function (params) { tooltipEl.style.display = 'none'; });

    // --- 3. SPOTLIGHT LOGIC ---
    network.on("selectNode", function (params) {
        if (params.nodes.length == 1) {
            var selectedNodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(selectedNodeId);
            var connectedEdges = network.getConnectedEdges(selectedNodeId);

            var allNodes = nodes.get();
            var updatedNodes = [];
            for (var i = 0; i < allNodes.length; i++) {
                var node = allNodes[i];
                if (!node.originalColor) node.originalColor = node.color;
                
                if (node.id == selectedNodeId || connectedNodes.includes(node.id)) {
                    node.color = node.originalColor;
                    node.opacity = 1.0;
                } else {
                    node.color = 'rgba(80, 80, 80, 0.1)'; 
                    node.opacity = 0.1;
                }
                updatedNodes.push(node);
            }
            nodes.update(updatedNodes);

            var allEdges = edges.get();
            var updatedEdges = [];
            for (var i = 0; i < allEdges.length; i++) {
                var edge = allEdges[i];
                if (!edge.originalColor) edge.originalColor = edge.color;
                
                if (connectedEdges.includes(edge.id)) {
                    edge.color = edge.originalColor;
                    edge.width = 4;
                    edge.font = { size: 14, color: "#ffffff", strokeWidth: 3, strokeColor: "#000000", background: "none" };
                } else {
                    edge.color = 'rgba(80, 80, 80, 0.05)'; 
                    edge.width = 1;
                    edge.font = { size: 0, color: "transparent" };
                }
                updatedEdges.push(edge);
            }
            edges.update(updatedEdges);
        }
    });

    network.on("deselectNode", function (params) {
        var allNodes = nodes.get();
        var updatedNodes = [];
        for (var i = 0; i < allNodes.length; i++) {
            var node = allNodes[i];
            if (node.originalColor) {
                node.color = node.originalColor;
                node.opacity = 1.0;
                updatedNodes.push(node);
            }
        }
        nodes.update(updatedNodes);

        var allEdges = edges.get();
        var updatedEdges = [];
        for (var i = 0; i < allEdges.length; i++) {
            var edge = allEdges[i];
            if (edge.originalColor) {
                edge.color = edge.originalColor;
                edge.width = edge.originalWidth || 2;
                edge.font = { size: 10, color: "#aaaaaa", strokeWidth: 0 };
                updatedEdges.push(edge);
            }
        }
        edges.update(updatedEdges);
    });
</script>
"""

def get_community_tags(G_full, community_nodes):
    topics = []
    keywords = []
    for node in community_nodes:
        for neighbor in G_full.neighbors(node):
            node_type = G_full.nodes[neighbor].get('type')
            if node_type == 'Topic': topics.append(neighbor)
            elif node_type == 'Keyword': keywords.append(neighbor)
    
    topic_counts = Counter(topics).most_common(1)
    main_topic = topic_counts[0][0] if topic_counts else "General"
    kw_str = ", ".join([k for k, v in Counter(keywords).most_common(3)])
    return main_topic, kw_str

def visualize_custom_tooltips():
    print(f"📂 Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print("❌ File not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    G_full = nx.node_link_graph(data)
    communities = list(community.greedy_modularity_communities(G_full))
    
    # FIX APPLIED HERE: select_menu=False, filter_menu=False
    net = Network(height="95vh", width="100%", bgcolor="#1a1b26", font_color="white", 
                  select_menu=False, filter_menu=False) 
    
    processed_notes = set()
    
    # 1. ADD NODES (Passing Data Only)
    for i, comm in enumerate(communities):
        comm_notes = [n for n in comm if G_full.nodes[n].get('type') == 'Note']
        if not comm_notes: continue
            
        topic_label, keywords_label = get_community_tags(G_full, comm_notes)
        color = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
        
        for note in comm_notes:
            processed_notes.add(note)
            
            net.add_node(
                note, 
                label=note,
                color=color, 
                size=25,
                shape="dot",
                group=f"Comm_{i}",
                # Pass custom data for JS to read
                custom_topic=topic_label,
                custom_keywords=keywords_label
            )

    # 2. ADD EDGES
    for u, v, data in G_full.edges(data=True):
        if u in processed_notes and v in processed_notes:
            relation = data.get('relation', 'related')
            label_text = ""
            if relation == "shares_keywords":
                shared = data.get("shared_kw", [])
                label_text = f"Shared: {', '.join(shared[:2])}"
                color, dashes = "#F1C40F", False
            elif relation == "semantic_keywords":
                matches = data.get("matches", "match")
                short_match = matches.split(',')[0].split('≈')[0].replace("'", "").strip()
                label_text = f"Semantic: {short_match}..."
                color, dashes = "#9B59B6", True
            elif relation == "content_similar":
                label_text = f"Sim: {data.get('similarity', 0):.2f}"
                color, dashes = "#3498DB", False
            else:
                label_text, color, dashes = relation, "#555555", False

            net.add_edge(u, v, title=label_text, label=label_text, color=color, width=2, dashes=dashes, 
                         font={'align': 'middle', 'color': '#aaaaaa', 'size': 10, 'strokeWidth': 0})

    # 3. PHYSICS
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": { "gravitationalConstant": -80, "springLength": 150, "springConstant": 0.04, "avoidOverlap": 0.5 },
        "minVelocity": 0.75, "solver": "forceAtlas2Based"
      },
      "interaction": { "hover": true } 
    }
    """)

    # 4. INJECT
    net.write_html(str(OUTPUT_FILE))
    
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Insert our custom JS/CSS before the body closes
    final_html = html_content.replace("</body>", JS_INTERACTIVITY + "\n</body>")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_html)

    abs_path = OUTPUT_FILE.resolve()
    print(f"\n✅ Visualization Ready: file://{abs_path}")

if __name__ == "__main__":
    visualize_custom_tooltips()
