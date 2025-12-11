import requests
import json
def mlc_ai_infer(prompt, server_url="http://localhost:8080/v1/completions", max_tokens=128, model="HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC"):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    response = requests.post(server_url, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["text"]

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

