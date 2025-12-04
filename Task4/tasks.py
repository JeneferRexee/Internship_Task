import os
import json
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# VALIDATION FUNCTIONS (used inside each stage)
# ------------------------------------------------------------

def validate_pdf_path(pdf_path):
    """Ensure the provided file exists and is a PDF."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("File is not a PDF.")
    return True


def validate_non_empty(data, error_msg):
    """Validates that extracted text/chunks/embeddings/etc. are not empty."""
    if not data:
        raise ValueError(error_msg)
    return True


# ------------------------------------------------------------
# 1. PDF TEXT EXTRACTION (using pdfplumber)
# ------------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)

    validate_non_empty(pages, "PDF extraction failed — no text detected.")
    return pages


# ------------------------------------------------------------
# 2. CHUNKING (paragraph-based)
# ------------------------------------------------------------

def chunk_text(pages_text):
    """Splits page text into paragraph-like chunks."""
    chunks = []
    for page in pages_text:
        parts = page.split("\n\n")
        for p in parts:
            cleaned = p.strip()
            if cleaned:
                chunks.append(cleaned)

    validate_non_empty(chunks, "Chunking failed — no valid chunks produced.")
    return chunks


# ------------------------------------------------------------
# 3. EMBEDDING MODEL LOADING + CHUNK EMBEDDINGS
# ------------------------------------------------------------

def generate_embeddings(chunks):
    """Generates embeddings for text chunks using MiniLM."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    validate_non_empty(embeddings, "Embedding generation failed.")
    return embeddings, model


# ------------------------------------------------------------
# 4. BUILD FAISS INDEX
# ------------------------------------------------------------

def build_faiss_index(embeddings):
    """Builds a FAISS index using L2 distance."""
    dim = embeddings.shape[1]            # Embedding dimensions
    index = faiss.IndexFlatL2(dim)       # Create index
    index.add(np.array(embeddings))      # Add all vectors into FAISS

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty after adding embeddings.")

    return index


# ------------------------------------------------------------
# 5. SEARCH QUERY USING FAISS
# ------------------------------------------------------------

def retrieve_relevant_chunks(query, model, index, chunks, k=5):
    """Embeds the query and retrieves top-k relevant text chunks."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)

    retrieved = [chunks[i] for i in indices[0]]
    validate_non_empty(retrieved, "FAISS did not return any relevant chunks.")

    return retrieved


# ------------------------------------------------------------
# 6. SAVE OUTPUT AS JSON
# ------------------------------------------------------------

def save_results(query, retrieved_chunks, output_path):
    """Saves retrieved chunks into a JSON file."""
    data = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    return output_path


# ------------------------------------------------------------
# MAIN PIPELINE (Single try/except here only)
# ------------------------------------------------------------

def process_pdf_rag_pipeline(pdf_path, query, output_json="retrieved_output.json"):
    """Full pipeline wrapped in a single error handler."""
    try:
        # Step 1: Validate and extract
        validate_pdf_path(pdf_path)
        pages_text = extract_text_from_pdf(pdf_path)

        # Step 2: Chunking
        chunks = chunk_text(pages_text)

        # Step 3: Embeddings
        embeddings, model = generate_embeddings(chunks)

        # Step 4: FAISS Index
        index = build_faiss_index(embeddings)

        # Step 5: Retrieval
        retrieved_chunks = retrieve_relevant_chunks(query, model, index, chunks)

        # Step 6: Save JSON
        save_path = save_results(query, retrieved_chunks, output_json)

        print(f"\n✔ SUCCESS — results saved to: {save_path}")
        return retrieved_chunks

    except Exception as e:
        print("\n PIPELINE FAILED:", e)


# ------------------------------------------------------------
# RUN EXAMPLE
# ------------------------------------------------------------
if __name__ == "__main__":
    pdf_path = r"C:\Users\Jenefer.RexeeGeorge\Documents\Internship_Task\Task4\uploads\AI_11_ISC_2 1.pdf"
    query = "Get the most recent vital signs"

    results = process_pdf_rag_pipeline(pdf_path, query)

    print("\nRetrieved Results:\n")
    if results:
        for r in results:
            print("•", r, "\n")
