from connections import find_connections, Note
from typing import Optional
import json

def build_graph(notes: list[Note]) -> dict:
    """Build complete graph with connections."""
    connections = find_connections(notes)
    
    # Filter low-strength connections (keep strong ones)
    strong_connections = [c for c in connections if c.strength > 0.2]
    
    return {
        "nodes": [
            {
                "id": n.id,
                "type": n.classification.get("note_type"),
                "key_idea": n.classification.get("key_idea"),
                "entities": n.classification.get("entities", [])
            }
            for n in notes
        ],
        "edges": [
            {
                "source": c.source,
                "target": c.target,
                "shared": c.shared_concepts,
                "strength": c.strength
            }
            for c in strong_connections
        ],
        "summary": {
            "total_notes": len(notes),
            "note_types": count_types([n.classification.get("note_type") for n in notes]),
            "total_entities": len(set(
                e for n in notes for e in n.classification.get("entities", [])
            )),
            "total_connections": len(strong_connections),
            "isolated_notes": len([n for n in notes if not any(
                e["source"] == n.id or e["target"] == n.id for e in strong_connections
            )])
        }
    }

def count_types(types: list[str]) -> dict:
    return {t: types.count(t) for t in set(types) if t}
