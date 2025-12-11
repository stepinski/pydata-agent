import os
import json

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

if USE_OLLAMA:
    import ollama
    def infer(prompt: str) -> str:
        response = ollama.generate(
            model="llama2",
            prompt=prompt,
            stream=False
        )
        return response['response']
else:
    from mlc_llm import MLCEngine
    engine = MLCEngine(model="Llama-2-7b-hf", device="auto")
    
    def infer(prompt: str) -> str:
        output = engine.generate(prompt, max_tokens=300)
        return output

def extract_json_from_response(response: str) -> dict:
    """Parse LLM response to JSON."""
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    return {
        "note_type": "other",
        "key_idea": response[:100],
        "confidence": 0.5,
        "entities": []
    }
