"""
Phase 2: FastAPI server wrapping our LLM.

Key idea: the model loads ONCE when the server starts (see the code
outside any function, at the bottom under __main__ / at import time).
Every request that comes in afterwards reuses that same loaded model.
This is exactly how real inference servers work.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import time

# ---------------------------------------------------------
# 1. Define what a request and response look like (schemas)
# ---------------------------------------------------------
# Pydantic checks incoming JSON automatically. If someone sends
# "max_tokens": "abc" instead of a number, FastAPI will reject it
# before your code even runs. This is "input validation" for free.

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


# ---------------------------------------------------------
# 3. Create the FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="LLM Inference Server")


@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is alive."""
    return {"status": "ok", "message": "Inference server is running"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """
    Takes a prompt, returns generated text + performance stats.
    This reuses the SAME model instance loaded above — no reloading.
    """
    start = time.time()

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
