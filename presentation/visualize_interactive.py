import json
import networkx as nx
from pyvis.network import Network
from pathlib import Path

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_FILE = Path("./output/knowledge_graph_enhanced.json")
# FIX: Wrap this in Path() so we can use .resolve() later
OUTPUT_FILE = Path("knowledge_graph_interactive.html")

# Aesthetic Palette (Tokyo Night Inspired)
COLORS = {
    "Topic":   {"background": "#2ECC71", "border": "#27AE60", "highlight": "#4cd183"},
    "Note":    {"background": "#3498DB", "border": "#2980B9", "highlight": "#5dade2"},
    "Keyword": {"background": "#E74C3C", "border": "#C0392B", "highlight": "#ec7063"},
    "Default": {"background": "#95A5A6", "border": "#7F8C8D", "highlight": "#aab7b8"}
}

# Edge Colors
EDGE_COLOR_DEFAULT = "#555555"
EDGE_COLOR_HIGHLIGHT = "#F1C40F" 

# ==========================================
# 🛠️ JAVASCRIPT INJECTION (The "Spotlight" Logic)
# ==========================================
JS_INTERACTIVITY = """
<script type="text/javascript">
    network.on("selectNode", function (params) {
        if (params.nodes.length == 1) {
            var selectedNodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(selectedNodeId);
            var connectedEdges = network.getConnectedEdges(selectedNodeId);

            // 1. DIM NODES
            var allNodes = nodes.get();
            var updatedNodes = [];
            for (var i = 0; i < allNodes.length; i++) {
                var node = allNodes[i];
                if (!node.originalColor) {
                    node.originalColor = node.color;
                }
                
                if (node.id == selectedNodeId || connectedNodes.includes(node.id)) {
                    node.color = node.originalColor; 
                } else {
                    node.color = 'rgba(200, 200, 200, 0.1)'; 
                }
                updatedNodes.push(node);
            }
            nodes.update(updatedNodes);

            // 2. DIM EDGES
            var allEdges = edges.get();
            var updatedEdges = [];
            for (var i = 0; i < allEdges.length; i++) {
                var edge = allEdges[i];
                if (!edge.originalColor) {
                    edge.originalColor = edge.color;
                }

                if (connectedEdges.includes(edge.id)) {
                    edge.color = edge.originalColor; 
                    edge.width = 3; 
                } else {
                    edge.color = 'rgba(200, 200, 200, 0.05)'; 
                    edge.width = 1;
                }
                updatedEdges.push(edge);
            }
            edges.update(updatedEdges);
        }
    });

    // RESTORE ON DESELECT
    network.on("deselectNode", function (params) {
        var allNodes = nodes.get();
        var updatedNodes = [];
        for (var i = 0; i < allNodes.length; i++) {
            var node = allNodes[i];
            if (node.originalColor) {
                node.color = node.originalColor;
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
                edge.width = edge.originalWidth || 1; 
                updatedEdges.push(edge);
            }
        }
        edges.update(updatedEdges);
    });
</script>
"""

# ==========================================
# 🏗️ BUILDER
# ==========================================
def build_interactive_graph():
    print(f"📂 Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print("❌ Data file not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    
    G = nx.node_link_graph(data)
    
    # 1. Apply Styles to NetworkX Graph
    for node, attrs in G.nodes(data=True):
        n_type = attrs.get("type", "Default")
        style = COLORS.get(n_type, COLORS["Default"])
        
        attrs["color"] = {
            "background": style["background"],
            "border": style["border"],
            "highlight": {
                "background": style["highlight"],
                "border": style["border"]
            }
        }
        attrs["borderWidth"] = 2
        
        if n_type == "Topic": attrs["size"] = 30
        elif n_type == "Note": attrs["size"] = 20
        else: attrs["size"] = 10

        attrs["title"] = f"<b>{n_type}:</b> {node}"

    # 2. Initialize PyVis
    net = Network(height="95vh", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)

    # 3. Configure Physics
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.09,
                "damping": 0.4,
                "avoidOverlap": 0.5
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        }
    }
    """)

    # 4. Generate Basic HTML
    print("⚡ Generating HTML...")
    # Convert Path to string for pyvis write_html
    net.write_html(str(OUTPUT_FILE))

    # 5. INJECT CUSTOM JAVASCRIPT
    print("🔧 Injecting 'Spotlight' interactivity code...")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()

    final_html = html_content.replace("</body>", JS_INTERACTIVITY + "\n</body>")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_html)

    # 6. Resolve Path (Now safe because OUTPUT_FILE is a Path object)
    abs_path = OUTPUT_FILE.resolve()
    print(f"\n✅ Interactive Graph Saved: file://{abs_path}")
    print("👉 Click any node to spotlight connections. Click empty space to reset.")

if __name__ == "__main__":
    build_interactive_graph()
