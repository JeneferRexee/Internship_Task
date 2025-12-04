import os
import json
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import PDF_PATH, OUTPUT_JSON

# ---------------- MANUAL LOGGING ---------------- #

LOG_FILE = "pipeline_log.txt"

def write_log(message):
    with open(LOG_FILE, "a") as log:
        log.write(message + "\n")

# ------------------------------------------------------------
# VALIDATION FUNCTIONS
# ------------------------------------------------------------

def validate_pdf_path(pdf_path):
    if not os.path.isfile(pdf_path):
        write_log(f"ERROR: PDF not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        write_log(f"ERROR: Not a PDF file: {pdf_path}")
        raise ValueError("Provided file is not a PDF.")
    write_log(f"PDF path validated: {pdf_path}")
    return True

def validate_non_empty(data, error_msg):
    if data is None or (hasattr(data, "__len__") and len(data) == 0) or (isinstance(data, np.ndarray) and data.size == 0):
        raise ValueError(error_msg)
    return True

# ------------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# ------------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    write_log(f"Extracting text from PDF: {pdf_path}")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)
            else:
                write_log(f"WARNING: Page {num+1} empty or unreadable.")
    validate_non_empty(pages, "PDF extraction failed — no text detected.")
    write_log(f"Extracted {len(pages)} pages with text.")
    return pages

# ------------------------------------------------------------
# 2. RECURSIVE CHUNKING
# ------------------------------------------------------------

def recursive_chunk(text, separators, max_len):
    text = text.strip()
    if len(text) <= max_len or not separators:
        return [text]
    sep = separators[0]
    parts = text.split(sep)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_len:
            chunks.extend(recursive_chunk(part, separators[1:], max_len))
        else:
            chunks.append(part)
    return chunks

def chunk_text(pages, max_len=400, separators=None):
    write_log("Starting text chunking...")
    if separators is None:
        separators = ["\n\n", "."]
    all_chunks = []
    for page_number, page_text in enumerate(pages, start=1):
        page_chunks = recursive_chunk(page_text, separators, max_len)
        all_chunks.extend([{"text": c, "page": page_number} for c in page_chunks if c.strip()])
    validate_non_empty(all_chunks, "Chunking failed — no valid chunks produced.")
    write_log(f"Created {len(all_chunks)} chunks.")
    return all_chunks

# ------------------------------------------------------------
# 3. EMBEDDING GENERATION (Sentence-Transformers)
# ------------------------------------------------------------

def generate_embeddings(chunks):
    write_log("Generating embeddings using MiniLM ('all-MiniLM-L6-v2')")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    validate_non_empty(embeddings, "Embedding generation failed using Sentence-Transformers.")
    write_log(f"Generated {len(embeddings)} embeddings.")
    return embeddings, model

# ------------------------------------------------------------
# 4. FAISS INDEX
# ------------------------------------------------------------

def build_faiss_index(embeddings):
    write_log("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    if index.ntotal == 0:
        write_log("ERROR: FAISS index empty.")
        raise ValueError("FAISS index is empty.")
    write_log(f"FAISS index built with {index.ntotal} items.")
    return index

# ------------------------------------------------------------
# 5. SEARCH (Sentence-Transformers Embeddings)
# ------------------------------------------------------------

def retrieve_relevant_chunks(query, model, index, chunks, k=5):
    write_log(f"Embedding query: {query}")
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    retrieved = [chunks[i] for i in indices[0]]
    validate_non_empty(retrieved, "FAISS returned no relevant chunks.")
    write_log(f"Retrieved {len(retrieved)} chunks for query.")
    return retrieved

# ------------------------------------------------------------
# 6. SAVE OUTPUT
# ------------------------------------------------------------

def save_results(query, retrieved_chunks, output_path):
    write_log("Saving results to JSON file.")
    data = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    write_log(f"Results saved to: {output_path}")
    return output_path

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def process_pdf_rag_pipeline(pdf_path, query, output_json=OUTPUT_JSON):
    write_log(f"--- RAG Pipeline Started ---")
    write_log(f"PDF: {pdf_path}")
    write_log(f"Query: {query}")
    print(query)
    try:
        validate_pdf_path(pdf_path)
        pages_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pages_text)
        embeddings, model = generate_embeddings(chunks)
        index = build_faiss_index(embeddings)
        retrieved_chunks = retrieve_relevant_chunks(query, model, index, chunks)
        save_path = save_results(query, retrieved_chunks, output_json)
        write_log("Pipeline completed successfully.\n")
        print(f"\n✔ SUCCESS — results saved at: {save_path}")
        return retrieved_chunks
    except Exception as e:
        write_log(f"PIPELINE FAILED: {e}\n")
        print("\nPIPELINE FAILED:", e)

# ------------------------------------------------------------
# RUN EXAMPLE
# ------------------------------------------------------------

if __name__ == "__main__":
    print("\nRAG started\n")
    pdf_path = os.path.join(PDF_PATH, "AI_11_ISC_2 1.pdf")
    query = "What is the medication reports"
    results = process_pdf_rag_pipeline(pdf_path, query)
    print("\nRetrieved Results:\n")
    if results:
        for r in results:
            print("•", r, "\n")
