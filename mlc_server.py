"""
MLC-AI Server: Start local LLM inference server
Required: mlc-llm package installed
Command: python mlc_server.py
"""

from mlc_llm import MLCEngine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SERVER_HOST = os.getenv("MLC_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("MLC_PORT", 8080))
MODEL_ID = os.getenv("MLC_MODEL", "HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC")
DEVICE = os.getenv("MLC_DEVICE", "auto")

# ==============================================================================
# INITIALIZE LLM ENGINE
# ==============================================================================

print(f"\n🚀 Initializing MLC-AI Engine...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {DEVICE}")
print(f"   Loading... (this may take 30-60 seconds on first run)\n")

try:
    engine = MLCEngine(model=MODEL_ID, device=DEVICE)
    print(f"✅ MLC-AI Engine ready!\n")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    print(f"\n   Make sure MLM-AI is installed:")
    print(f"   pip install mlc-llm")
    raise

# ==============================================================================
# FASTAPI SERVER
# ==============================================================================

app = FastAPI(title="MLC-AI Local LLM Server", version="1.0")


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response"""
    model: str
    choices: list
    usage: dict


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "endpoint": "/v1/completions"
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    """
    OpenAI-compatible completions endpoint
    
    Example:
    POST /v1/completions
    {
        "model": "HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC",
        "prompt": "What is machine learning?",
        "max_tokens": 128
    }
    """
    try:
        # Generate completion using MLC-AI
        output = engine.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Return OpenAI-compatible response
        return CompletionResponse(
            model=request.model,
            choices=[
                {
                    "text": output,
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(request.prompt.split()) + len(output.split())
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """
    OpenAI-compatible chat completions endpoint (simplified)
    Converts messages to prompt format
    """
    try:
        # Convert messages to single prompt
        prompt = ""
        for msg in request.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        
        # Generate using MLC-AI
        output = engine.generate(
            prompt,
            max_tokens=request.get("max_tokens", 128)
        )
        
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": output
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/v1/models")
async def list_models():
    """List available models"""
    return {
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local"
            }
        ]
    }


@app.get("/")
async def root():
    """Root endpoint - API documentation"""
    return {
        "name": "MLC-AI Local LLM Server",
        "version": "1.0",
        "status": "running",
        "model": MODEL_ID,
        "endpoints": {
            "health": "GET /health",
            "completions": "POST /v1/completions",
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models"
        },
        "documentation": "Send requests to /v1/completions endpoint"
    }


# ==============================================================================
# STARTUP & SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    print(f"\n{'='*70}")
    print(f"🌐 MLC-AI Server Starting")
    print(f"{'='*70}")
    print(f"Host: {SERVER_HOST}")
    print(f"Port: {SERVER_PORT}")
    print(f"Model: {MODEL_ID}")
    print(f"\nReady for requests!")
    print(f"Health check: http://{SERVER_HOST}:{SERVER_PORT}/health")
    print(f"Completions: POST http://{SERVER_HOST}:{SERVER_PORT}/v1/completions")
    print(f"{'='*70}\n")


@app.on_event("shutdown")
async def shutdown_event():
    print(f"\n🛑 MLC-AI Server shutting down...")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 Starting MLC-AI Local LLM Server")
    print(f"{'='*70}\n")
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )
