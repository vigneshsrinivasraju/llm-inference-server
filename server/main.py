"""
Phase 10: Server with API key auth, rate limiting, metrics, and RAG.
UI ("/") and health check stay public. /metrics is also public for demo purposes.
"""

from fastapi import FastAPI, Depends, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from llama_cpp import Llama
import time
import json
import threading
import os
import urllib.request

from server.auth import verify_api_key
from server.rate_limit import check_rate_limit
from server.metrics import log_request, get_metrics_summary
from server.rag import extract_text, chunk_text, find_relevant_chunks, build_rag_prompt

MODEL_PATH = "models/tinyllama.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_taken_sec: float
    tokens_per_second: float


class RagResponse(BaseModel):
    answer: str
    chunks_used: int
    tokens_generated: int
    time_taken_sec: float


os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Model not found locally - downloading (first startup only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

print("Loading model... please wait")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)
print("Model loaded. Server ready.")

inference_lock = threading.Lock()

app = FastAPI(title="LLM Inference Server")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Inference server is running"}


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/metrics")
def metrics():
    return get_metrics_summary()


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, auth: dict = Depends(verify_api_key)):
    api_key = auth["api_key"]
    try:
        check_rate_limit(auth)
    except Exception as e:
        log_request(api_key, "/generate", 429, 0)
        raise e

    start = time.time()
    try:
        with inference_lock:
            output = llm(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=["</s>"]
            )
    except Exception as e:
        elapsed = time.time() - start
        log_request(api_key, "/generate", 500, elapsed)
        raise e

    elapsed = time.time() - start
    generated_text = output["choices"][0]["text"].strip()
    tokens_generated = output["usage"]["completion_tokens"]
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    log_request(api_key, "/generate", 200, elapsed, tokens_generated)

    return GenerateResponse(
        text=generated_text,
        tokens_generated=tokens_generated,
        time_taken_sec=round(elapsed, 3),
        tokens_per_second=round(tokens_per_sec, 2)
    )


def token_generator(prompt: str, max_tokens: int, temperature: float, api_key: str):
    start = time.time()
    token_count = 0

    with inference_lock:
        stream = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["</s>"], stream=True)
        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            if piece:
                token_count += 1
                yield f"data: {json.dumps({'token': piece})}\n\n"

    elapsed = time.time() - start
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
    log_request(api_key, "/stream", 200, elapsed, token_count)

    final_payload = json.dumps({
        "done": True,
        "tokens_generated": token_count,
        "time_taken_sec": round(elapsed, 3),
        "tokens_per_second": round(tokens_per_sec, 2)
    })
    yield f"data: {final_payload}\n\n"


@app.post("/stream")
def stream(request: GenerateRequest, auth: dict = Depends(verify_api_key)):
    api_key = auth["api_key"]
    try:
        check_rate_limit(auth)
    except Exception as e:
        log_request(api_key, "/stream", 429, 0)
        raise e

    return StreamingResponse(
        token_generator(request.prompt, request.max_tokens, request.temperature, api_key),
        media_type="text/event-stream"
    )


@app.post("/rag/ask", response_model=RagResponse)
async def rag_ask(
    file: UploadFile = File(...),
    question: str = Form(...),
    auth: dict = Depends(verify_api_key)
):
    """
    Accepts a file upload (PDF or TXT) + a question about it.
    Runs the full RAG pipeline: extract -> chunk -> embed -> retrieve -> generate.
    """
    api_key = auth["api_key"]
    try:
        check_rate_limit(auth)
    except Exception as e:
        log_request(api_key, "/rag/ask", 429, 0)
        raise e

    start = time.time()

    file_bytes = await file.read()
    raw_text = extract_text(file_bytes, file.filename)
    chunks = chunk_text(raw_text)
    relevant_chunks = find_relevant_chunks(question, chunks, top_k=3)
    print(f"DEBUG - Retrieved chunks: {relevant_chunks}")
    rag_prompt = build_rag_prompt(question, relevant_chunks)
    print(f"DEBUG - Full prompt sent to model:\n{rag_prompt}")

    with inference_lock:
        output = llm(
            rag_prompt,
            max_tokens=200,
            temperature=0.1,  # very low temperature - maximize determinism for factual extraction
            stop=["</s>"]
        )

    elapsed = time.time() - start
    answer = output["choices"][0]["text"].strip()
    tokens_generated = output["usage"]["completion_tokens"]

    log_request(api_key, "/rag/ask", 200, elapsed, tokens_generated)

    return RagResponse(
        answer=answer,
        chunks_used=len(relevant_chunks),
        tokens_generated=tokens_generated,
        time_taken_sec=round(elapsed, 3)
    )


@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc):
    log_request("unknown", request.url.path, 401, 0)
    return JSONResponse(status_code=401, content={"detail": str(exc.detail)})
