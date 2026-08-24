"""
Phase 2 + 3 (thread-safety fix): FastAPI server wrapping our LLM.

IMPORTANT LESSON:
llama-cpp-python's basic Llama object is NOT safe to call from multiple
threads at once - doing so causes crashes/corruption, which is exactly
what we saw when benchmarking at concurrency=5.

Real inference engines (vLLM, TGI, Triton) solve this properly with
continuous batching, letting many requests share GPU compute safely
and efficiently at the same time.

We don't have that here (CPU + llama.cpp), so instead we use a LOCK:
a mechanism that forces only ONE request to actually run inference at
a time, while others wait their turn in a queue. This keeps us safe
and correct, and honestly demonstrates the real difference between
naive serving and proper batched serving - which is worth discussing
directly in your README.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
import time
import json
import threading

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
# 2. Load the model ONCE, when the server starts
# ---------------------------------------------------------
print("Loading model... please wait")
llm = Llama(
    model_path="models/tinyllama.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("Model loaded. Server ready.")

# This lock ensures only one thread can run inference at a time.
# Every request must "acquire" this lock before calling the model,
# and releases it when done - forcing safe, one-at-a-time access.
inference_lock = threading.Lock()


# ---------------------------------------------------------
# 3. FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="LLM Inference Server")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Inference server is running"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Non-streaming: waits for the full response, then returns it."""
    start = time.time()

    # Acquire the lock before touching the model. If another request
    # is currently generating, this line will simply WAIT here until
    # the lock is free - that's the "queueing" behavior in action.
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
    """Generator for streaming - same lock protection applies here."""
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
    """Streaming endpoint: returns tokens one at a time as they're generated."""
    return StreamingResponse(
        token_generator(request.prompt, request.max_tokens, request.temperature),
        media_type="text/event-stream"
    )
