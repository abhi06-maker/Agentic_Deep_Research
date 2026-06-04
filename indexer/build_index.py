"""
Parse PDFs → chunk text → build hybrid index (BM25 + FAISS dense).
Saves chunks to data/chunks.jsonl and FAISS index to data/faiss_index/
"""
import fitz  # pymupdf
import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) == 0:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:  # skip tiny chunks
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def find_pdf_path(arxiv_id):
    """Try multiple filename formats to find the PDF."""
    # Primary format: replace . and / with _
    base = arxiv_id.replace("/", "_").replace(".", "_")
    
    candidates = [
        os.path.join(PDF_DIR, base + ".pdf"),                    # 2606_03331v1.pdf
        os.path.join(PDF_DIR, arxiv_id.replace("/", "_") + ".pdf"),  # 2606.03331v1.pdf
        os.path.join(PDF_DIR, arxiv_id + ".pdf"),                # 2606.03331v1.pdf
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def build_index():
    # Load paper metadata
    papers = {}
    with open(os.path.join(DATA_DIR, "papers.jsonl"), encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p

    print(f"📚 Loaded {len(papers)} papers")

    all_chunks = []
    pdf_found = 0
    abstract_only = 0

    print("📄 Parsing PDFs and chunking...")

    for arxiv_id, paper in tqdm(papers.items()):
        pdf_path = find_pdf_path(arxiv_id)
        text = ""

        if pdf_path:
            text = extract_text_from_pdf(pdf_path)
            if len(text.split()) >= 100:
                pdf_found += 1
            else:
                text = ""  # bad PDF, fall back

        if len(text.split()) < 100:
            # Use title + abstract as fallback
            text = paper.get("title", "") + ". " + paper.get("abstract", "")
            abstract_only += 1

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{arxiv_id}__chunk_{i}",
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "text": chunk,
                "chunk_index": i,
            })

    print(f"\n📊 Chunking stats:")
    print(f"   PDFs parsed:      {pdf_found}")
    print(f"   Abstract-only:    {abstract_only}")
    print(f"   Total chunks:     {len(all_chunks)}")

    if len(all_chunks) == 0:
        print("❌ No chunks created! Check your data/pdfs/ folder.")
        return

    # Save chunks
    chunks_path = os.path.join(DATA_DIR, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"💾 Saved chunks to {chunks_path}")

    # ----- BM25 Index -----
    print("\n📊 Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(DATA_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print("   ✅ BM25 saved to data/bm25.pkl")

    # ----- Dense FAISS Index -----
    print("\n🧠 Encoding chunks with sentence-transformers (this takes a few minutes)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in all_chunks]
    batch_size = 128
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")

    faiss_dir = os.path.join(DATA_DIR, "faiss_index")
    os.makedirs(faiss_dir, exist_ok=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(faiss_dir, "index.faiss"))
    np.save(os.path.join(faiss_dir, "embeddings.npy"), embeddings)

    print(f"\n✅ Index built!")
    print(f"   FAISS shape: {embeddings.shape}")
    print(f"   Saved to data/faiss_index/")

if __name__ == "__main__":
    build_index()