from pydantic import BaseModel
from mlc_server_client import mlc_ai_infer, extract_json_from_response

class NoteClassification(BaseModel):
    note_type: str
    key_idea: str
    confidence: float
    entities: list[str]

def classify_with_local_llm(chunk: str) -> NoteClassification:
    prompt = f"""Analyze this note. Respond ONLY with valid JSON.

NOTE:
{chunk[:800]}

{{"note_type": "meeting|idea|reference|todo|other", 
  "key_idea": "1-2 sentence summary",
  "confidence": 0.5,
  "entities": ["concept1", "concept2"]}}

JSON:""" 
    
    response = mlc_ai_infer(prompt)
    #  response = infer(prompt)
    parsed = extract_json_from_response(response)
    
    return NoteClassification(
        note_type=parsed.get("note_type", "other"),
        key_idea=parsed.get("key_idea", chunk[:100]),
        confidence=float(parsed.get("confidence", 0.5)),
        entities=parsed.get("entities", [])
    )
