from typing import Optional
from pydantic import BaseModel

class Note(BaseModel):
    id: str
    chunk: str
    classification: dict

class Connection(BaseModel):
    source: str
    target: str
    shared_concepts: list[str]
    strength: float  # 0-1 based on overlap

def find_connections(notes: list[Note]) -> list[Connection]:
    """Find connections based on shared entities and concepts."""
    connections = []
    
    for i, note_a in enumerate(notes):
        for note_b in notes[i+1:]:
            entities_a = set(note_a.classification.get("entities", []))
            entities_b = set(note_b.classification.get("entities", []))
            
            # Shared entities
            shared = entities_a & entities_b
            
            # Same note type (thematic connection)
            same_type = note_a.classification.get("note_type") == note_b.classification.get("note_type")
            
            # Create connection if meaningful
            if shared or same_type:
                strength = len(shared) / max(len(entities_a), len(entities_b), 1)
                shared_list = list(shared) if shared else ["same_type"]
                
                connections.append(Connection(
                    source=note_a.id,
                    target=note_b.id,
                    shared_concepts=shared_list,
                    strength=max(strength, 0.3 if same_type else 0)
                ))
    
    return connections
