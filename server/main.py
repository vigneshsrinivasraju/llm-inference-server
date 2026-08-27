"""
Deployment version: FastAPI server wrapping our LLM, now serving a
simple web UI at "/" in addition to the existing API endpoints.

Designed to run on Hugging Face Spaces (free CPU tier):
  - On startup, downloads the model automatically if it's not already
    present (Spaces containers start fresh each build).
  - Serves static/index.html at the root URL so visitors get a working
    chat UI instead of a bare JSON health check.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
import time
import json
import threading
import os
import urllib.request

MODEL_PATH = "models/tinyllama.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# ---------------------------------------------------------
# 1. Request/response schemas
# ---------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_taken_sec: float
    tokens_per_second: float


# ---------------------------------------------------------
# 2. Download the model if it's not already there, then load it
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Model not found locally - downloading (first startup only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

print("Loading model... please wait")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("Model loaded. Server ready.")

inference_lock = threading.Lock()


# ---------------------------------------------------------
# 3. FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="LLM Inference Server")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Inference server is running"}


@app.get("/")
def serve_ui():
    """Serves the chat UI at the root URL - this is what visitors see."""
    return FileResponse("static/index.html")


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    start = time.time()

    with inference_lock:
        output = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["</s>"]
        )

    elapsed = time.time() - start

    generated_text = output["choices"][0]["text"].strip()
    tokens_generated = output["usage"]["completion_tokens"]
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    return GenerateResponse(
        text=generated_text,
        tokens_generated=tokens_generated,
        time_taken_sec=round(elapsed, 3),
        tokens_per_second=round(tokens_per_sec, 2)
    )


def token_generator(prompt: str, max_tokens: int, temperature: float):
    start = time.time()
    token_count = 0

    with inference_lock:
        stream = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>"],
            stream=True
        )

        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            if piece:
                token_count += 1
                payload = json.dumps({"token": piece})
                yield f"data: {payload}\n\n"

    elapsed = time.time() - start
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

    final_payload = json.dumps({
        "done": True,
        "tokens_generated": token_count,
        "time_taken_sec": round(elapsed, 3),
        "tokens_per_second": round(tokens_per_sec, 2)
    })
    yield f"data: {final_payload}\n\n"


@app.post("/stream")
def stream(request: GenerateRequest):
    return StreamingResponse(
        token_generator(request.prompt, request.max_tokens, request.temperature),
        media_type="text/event-stream"
    )
