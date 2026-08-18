"""
Phase 0 sanity check.
This just proves: model loads + generates text on CPU.
Nothing fancy yet — no API, no streaming. Just the raw engine working.
"""

from llama_cpp import Llama
import time

# Load the model into memory
# n_ctx = how many tokens of context the model can "remember" at once
# n_threads = how many CPU cores to use for generation
print("Loading model... (this takes a few seconds)")
llm = Llama(
    model_path="models/tinyllama.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("Model loaded.\n")

prompt = "Explain what a neural network is in two sentences."

start = time.time()

output = llm(
    prompt,
    max_tokens=100,
    temperature=0.7,
    stop=["</s>"]  # tells the model when to stop generating
)

elapsed = time.time() - start

generated_text = output["choices"][0]["text"]
tokens_generated = output["usage"]["completion_tokens"]

print("PROMPT:", prompt)
print("\nRESPONSE:", generated_text.strip())
print(f"\n--- Stats ---")
print(f"Tokens generated: {tokens_generated}")
print(f"Time taken: {elapsed:.2f}s")
print(f"Tokens/sec: {tokens_generated / elapsed:.2f}")
