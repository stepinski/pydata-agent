
from extract import extract_pdf_text, chunk_text
from agent import classify_with_local_llm
from graph import build_graph, Note
from config import DATA_DIR, OUTPUT_DIR
import json
from pathlib import Path

def process_notes(pdf_folder: Path = DATA_DIR):
    pdf_files = list(pdf_folder.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")
    
    all_notes = []
    note_id = 0
    
    for pdf_file in pdf_files:
        print(f"\n📄 Processing {pdf_file.name}...")
        text = extract_pdf_text(str(pdf_file))
        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunks")
        
        for chunk in chunks:
            classification = classify_with_local_llm(chunk)
            note = Note(
                id=f"note_{note_id}",
                chunk=chunk,
                classification=classification.model_dump()
            )
            all_notes.append(note)
            note_id += 1
            print(f"    ✓ {classification.note_type}")
    
    # Build graph
    print(f"\n🔗 Building knowledge graph...")
    graph = build_graph(all_notes)
    
    # Save outputs
    with open(OUTPUT_DIR / "graph.json", "w") as f:
        json.dump(graph, f, indent=2)
    
    print(f"\n✅ Graph: {graph['summary']['total_notes']} notes, {graph['summary']['total_connections']} connections")
    print(f"   Isolated notes: {graph['summary']['isolated_notes']}")
    return graph

if __name__ == "__main__":
    graph = process_notes()
    print(json.dumps(graph["summary"], indent=2))
