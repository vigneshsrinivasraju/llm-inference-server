"""
Phase 10: RAG (Retrieval-Augmented Generation) pipeline for user-uploaded documents.

Flow per request:
  1. Extract raw text from the uploaded file (PDF or TXT)
  2. Split the text into overlapping chunks
  3. Embed each chunk into a vector (using a small local model)
  4. Embed the user's question the same way
  5. Find the chunks most similar to the question (cosine similarity)
  6. Return the top chunks as context, ready to inject into a prompt

Everything happens in-memory, per request - no persistent vector
database needed at this scale, since each upload is a fresh document.
"""

import io
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Loaded ONCE at import time (same "load once, reuse many times" pattern
# as the LLM itself) - this is a small, fast embedding model that runs
# comfortably on CPU.
print("Loading embedding model for RAG...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extracts raw text from an uploaded PDF or TXT file's bytes."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        # Treat anything else as plain text
        text = file_bytes.decode("utf-8", errors="ignore")
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into overlapping chunks of `chunk_size` characters.
    Overlap helps avoid losing context that straddles a chunk boundary.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def find_relevant_chunks(question: str, chunks: list, top_k: int = 3) -> list:
    """
    Embeds the question and all chunks, then returns the top_k chunks
    most semantically similar to the question (via cosine similarity).
    """
    if not chunks:
        return []

    chunk_embeddings = embedding_model.encode(chunks)
    question_embedding = embedding_model.encode([question])[0]

    # Cosine similarity between the question and every chunk
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
    )

    # Get indices of the top_k highest-similarity chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


def build_rag_prompt(question: str, relevant_chunks: list) -> str:
    """
    Combines retrieved context with the user's question into TinyLlama's
    expected chat format. Small models like TinyLlama follow instructions
    much more reliably when given their specific trained chat template -
    plain text prompts often get ignored in favor of the model's own
    (sometimes wrong) memorized knowledge.
    """
    context = "\n\n".join(relevant_chunks)
    return f"""<|system|>
You are a precise assistant. Find the EXACT sentence in the context that answers the question, then answer using ONLY that sentence. Do not combine numbers from different sentences. If the answer is not in the context, say "I cannot find this information in the document."</s>
<|user|>
Context:
{context}

Question: {question}

First, quote the single sentence from the context that contains the answer. Then give a one-sentence final answer.</s>
<|assistant|>
"""
