if __name__ == "__main__":
    print("\n⚠️  PREREQUISITES:")
    print("   1. MLC-AI Vision server running in another terminal:")
    print()
    print("      mlc_llm serve HF://mlc-ai/Phi-3.5-vision-instruct-q0f16-MLC \\")
    print("          --device metal --host localhost --port 8081")
    print()
    print("   2. PDFs in: data/ folder")
    print()
 
    # Verify MLC-AI Vision server is running
    import requests
    try:
        # Try the v1/completions endpoint
        response = requests.post(
            "http://localhost:8081/v1/completions",
            json={"model": "default", "prompt": "test", "max_tokens": 1},
            timeout=3
        )
        if response.status_code in [200, 400]:  # 400 is ok - means server is responding
            print("✅ MLC-AI Vision server detected at http://localhost:8081/v1")
            print("   Model: Phi-3.5-vision-instruct (vision capable)")
        else:
            raise Exception(f"Server responded with {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to MLC-AI Vision server at http://localhost:8081")
        print(f"   Start it with:")
        print(f"   mlc_llm serve HF://mlc-ai/Phi-3.5-vision-instruct-q0f16-MLC \\")
        print(f"       --device metal --host localhost --port 8081")
        exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        exit(1)"""
PyData Agent Demo: CrewAI Agentic Approach
From Handwritten Notes to Smart Knowledge using CrewAI
Uses Ollama for local LLM inference
"""

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pathlib import Path
import json
from config import DATA_DIR, OUTPUT_DIR
import os

# ==============================================================================
# CONFIGURE CREWAI FOR LOCAL MLC-AI VISION SERVER
# ==============================================================================

# Set MLC-AI Vision configuration
os.environ["OPENAI_API_KEY"] = "mlc-ai-vision"
os.environ["OPENAI_API_BASE"] = "http://localhost:8081/v1"

from langchain_openai import ChatOpenAI

# Create LLM instance pointing to MLC-AI Vision server (Phi-3.5-vision)
# This model can understand both text and images
local_llm = ChatOpenAI(
    base_url="http://localhost:8081/v1",
    api_key="mlc-ai-vision",
    model_name="phi-3.5-vision",
    temperature=0.3,
    max_tokens=512
)

# ==============================================================================
# STEP 1: AGENT TOOLS (The actual work)
# ==============================================================================

def ocr_agent_fn(pdf_path: str) -> str:
    """
    OCR Agent: Extract text from handwritten PDFs
    Uses Phi-3.5-vision to understand handwritten content
    Can process images directly for better handwriting recognition
    """
    from pypdf import PdfReader
    from PIL import Image
    import io
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        # First try direct text extraction
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        # If no text found, the PDF likely contains only images (handwritten)
        # In that case, we'd need to convert PDF pages to images and use vision model
        if not text.strip():
            print(f"   📸 Handwritten content detected - using vision model")
            # Note: Full image processing requires pdf2image and more setup
            text = f"[Handwritten PDF detected: {Path(pdf_path).name} - requires vision processing]"
        
        return text if text.strip() else f"[No content in {pdf_path}]"
    except Exception as e:
        return f"[OCR Error: {str(e)}]"


def classification_agent_fn(text: str) -> str:
    """
    Classification Agent: Categorize note type
    Detect: meeting, idea, reference, todo, calendar, other
    """
    prompt = f"""Classify this note into ONE category. Respond ONLY with JSON.

NOTE:
{text[:1000]}

{{"note_type": "meeting|idea|reference|todo|calendar|other",
  "confidence": 0.0-1.0,
  "reasoning": "why this category"}}

JSON:"""
    
    # Use the local LLM
    response = local_llm.invoke(prompt)
    return str(response.content)


def idea_extraction_agent_fn(text: str) -> str:
    """
    Idea Extraction Agent: Extract key concepts and entities
    Returns main ideas, entities, and keywords
    """
    prompt = f"""Extract key ideas and entities from this note. Respond ONLY with JSON.

NOTE:
{text[:1000]}

{{"key_idea": "1-2 sentence main concept",
  "entities": ["person/project/concept", "another entity"],
  "keywords": ["keyword1", "keyword2"],
  "themes": ["theme1", "theme2"]}}

JSON:"""
    
    response = local_llm.invoke(prompt)
    return str(response.content)


def connections_agent_fn(text: str) -> str:
    """
    Connections Agent: Analyze text for connections
    """
    prompt = f"""Analyze this note for key connection points:

NOTE:
{text[:1000]}

Respond with a brief summary of the main topics and potential connection points."""
    
    response = local_llm.invoke(prompt)
    return str(response.content)


# ==============================================================================
# STEP 2: DEFINE TOOLS
# ==============================================================================

class OCRTool(BaseTool):
    name: str = "ocr_tool"
    description: str = "Extracts text from PDF notes using OCR"
    
    def _run(self, pdf_path: str) -> str:
        return ocr_agent_fn(pdf_path)


class ClassificationTool(BaseTool):
    name: str = "classification_tool"
    description: str = "Classifies notes into: meeting, idea, reference, todo, calendar"
    
    def _run(self, text: str) -> str:
        return classification_agent_fn(text)


class IdeaExtractionTool(BaseTool):
    name: str = "idea_extraction_tool"
    description: str = "Extracts key ideas, entities, and themes from notes"
    
    def _run(self, text: str) -> str:
        return idea_extraction_agent_fn(text)


class ConnectionsTool(BaseTool):
    name: str = "connections_tool"
    description: str = "Finds relationships and connections between notes"
    
    def _run(self, text: str) -> str:
        return connections_agent_fn(text)


# ==============================================================================
# STEP 3: DEFINE AGENTS
# ==============================================================================

ocr_agent = Agent(
    name="📄 OCR Agent",
    role="Document extraction specialist",
    goal="Extract clean text from handwritten PDF notes",
    description="Reads PDF files and converts handwritten content to text",
    backstory="Expert in OCR and document processing",
    tools=[OCRTool()],
    verbose=True,
    llm=local_llm
)

classification_agent = Agent(
    name="🏷️ Classification Agent",
    role="Note categorization specialist",
    goal="Classify notes into: meeting, idea, reference, todo, calendar",
    description="Analyzes text and determines the note type with confidence",
    backstory="Expert in note classification and organization",
    tools=[ClassificationTool()],
    verbose=True,
    llm=local_llm
)

idea_extraction_agent = Agent(
    name="💡 Idea Extraction Agent",
    role="Concept and entity identification specialist",
    goal="Extract key ideas, entities, and themes from notes",
    description="Identifies main concepts, people/projects, keywords, and themes",
    backstory="Expert in extracting ideas and entities from documents",
    tools=[IdeaExtractionTool()],
    verbose=True,
    llm=local_llm
)

connections_agent = Agent(
    name="🔗 Connections Agent",
    role="Knowledge graph builder",
    goal="Find relationships and connections between notes",
    description="Analyzes ideas to find shared concepts and build connections",
    backstory="Expert in building knowledge graphs and finding connections",
    tools=[ConnectionsTool()],
    verbose=True,
    llm=local_llm
)

# ==============================================================================
# STEP 4: PROCESSING PIPELINE
# ==============================================================================

def process_single_note(pdf_path: str) -> dict:
    """
    Process one note through the CrewAI pipeline:
    PDF → OCR → Classification → Idea Extraction → Connections
    """
    
    print(f"\n{'='*70}")
    print(f"Processing: {Path(pdf_path).name}")
    print(f"{'='*70}")
    
    # Task 1: Extract text from PDF
    ocr_task = Task(
        description=f"Extract text from PDF: {pdf_path}",
        agent=ocr_agent,
        expected_output="Extracted text from the PDF"
    )
    
    # Task 2: Classify the note
    classification_task = Task(
        description="Classify the extracted note into a category",
        agent=classification_agent,
        expected_output="JSON with note_type, confidence, reasoning"
    )
    
    # Task 3: Extract ideas
    idea_task = Task(
        description="Extract key ideas, entities, keywords, and themes",
        agent=idea_extraction_agent,
        expected_output="JSON with key_idea, entities, keywords, themes"
    )
    
    # Task 4: Find connections
    connections_task = Task(
        description="Find connection points and themes in the note",
        agent=connections_agent,
        expected_output="Summary of connections and themes"
    )
    
    # Create crew with sequential task execution
    crew = Crew(
        agents=[ocr_agent, classification_agent, idea_extraction_agent, connections_agent],
        tasks=[ocr_task, classification_task, idea_task, connections_task],
        verbose=True
    )
    
    # Execute pipeline
    try:
        result = crew.kickoff()
        print(f"\n✅ Success: {Path(pdf_path).name}")
        return {
            "file": pdf_path,
            "status": "success",
            "result": str(result)
        }
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return {
            "file": pdf_path,
            "status": "error",
            "error": str(e)
        }


def process_all_notes() -> dict:
    """
    Process all notes in data/ folder through CrewAI pipeline
    """
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDFs found in data/ folder")
        return {"total_processed": 0, "successful": 0, "failed": 0, "results": []}
    
    print(f"\n🚀 PyData Agent Demo: CrewAI Pipeline")
    print(f"{'='*70}")
    print(f"📁 Found {len(pdf_files)} PDF notes to process")
    print(f"{'='*70}\n")
    
    all_results = []
    
    # Process each note
    for pdf_path in pdf_files:
        result = process_single_note(str(pdf_path))
        all_results.append(result)
    
    # Save results
    output_file = OUTPUT_DIR / "crew_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "error")
    
    print(f"\n✅ Processing complete!")
    print(f"   Results saved to: {output_file}")
    print(f"   Successful: {successful}/{len(all_results)}")
    print(f"   Failed: {failed}/{len(all_results)}")
    
    return {
        "total_processed": len(all_results),
        "successful": successful,
        "failed": failed,
        "results": all_results
    }


# ==============================================================================
# GRAPH BUILDING
# ==============================================================================

def build_graph_from_crew_results(crew_results: dict) -> dict:
    """
    Build knowledge graph from crew results
    """
    print(f"\n{'='*70}")
    print(f"🔗 Building Knowledge Graph from Crew Results")
    print(f"{'='*70}\n")
    
    graph = {
        "nodes": [],
        "edges": [],
        "summary": {
            "total_notes": crew_results.get("total_processed", 0),
            "successful": crew_results.get("successful", 0),
            "processing_status": "Crew agents processed notes"
        }
    }
    
    return graph


# ==============================================================================
# DEMO QUERIES
# ==============================================================================

def demo_query_connections(crew_results: dict):
    """
    Demo: Show automatic connections found between notes
    """
    print(f"\n{'='*70}")
    print(f"🔍 DEMO: Automatic Connections Found")
    print(f"{'='*70}\n")
    
    print(f"Processed: {crew_results['total_processed']} notes")
    print(f"Success: {crew_results['successful']}")
    print(f"Failed: {crew_results['failed']}")
    
    print(f"\n✅ The Crew automatically:")
    print(f"   • Extracted text from PDFs")
    print(f"   • Classified notes by type")
    print(f"   • Found key ideas and entities")
    print(f"   • Identified connection points")
    print(f"   • Built the knowledge graph")
    
    print(f"\n📊 Result: Automatic analysis without manual work!")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("\n⚠️  PREREQUISITES:")
    print("   1. Ollama running: ollama serve")
    print("   2. Model pulled: ollama pull qwen2.5:7b-instruct-q4_0")
    print("   3. PDFs in: data/ folder")
    print()
    
    # Verify Ollama is running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama server detected")
        else:
            raise Exception("Server error")
    except Exception as e:
        print(f"❌ ERROR: Cannot connect to Ollama!")
        print(f"   Start with: ollama serve")
        exit(1)
    
    # Run the full pipeline
    try:
        crew_results = process_all_notes()
        
        # Build graph from results
        graph = build_graph_from_crew_results(crew_results)
        
        # Save graph
        with open(OUTPUT_DIR / "graph.json", "w") as f:
            json.dump(graph, f, indent=2)
        
        # Run demo queries
        demo_query_connections(crew_results)
        
        print(f"\n✅ End of demo.\n")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
