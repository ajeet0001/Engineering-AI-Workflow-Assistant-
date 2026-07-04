"""
test_rag.py
-----------
Standalone test for the RAG pipeline (no API key required).
Tests: document loading, chunking, embedding, FAISS build, retrieval, persistence.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("=" * 55)
print("  RAG PIPELINE TEST (no API key required)")
print("=" * 55)

# Test 1: Load knowledge base
print("\n[Test 1] Loading knowledge base documents...")
from src.rag.loader import load_knowledge_base

docs = load_knowledge_base(Path("data/knowledge_base"))
assert len(docs) > 0, "No documents loaded!"
print(f"  => Loaded {len(docs)} document(s):")
for d in docs:
    print(f"     - {d.metadata['source']}  ({len(d.page_content):,} chars)")

# Test 2: Chunk documents
print("\n[Test 2] Chunking documents (size=500, overlap=50)...")
from src.rag.chunking import chunk_documents

chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
assert len(chunks) > 0, "No chunks created!"
print(f"  => Created {len(chunks)} chunk(s)")
sample = chunks[0].page_content[:100].strip().replace("\n", " ")
print(f"     Sample chunk: '{sample}'")

# Test 3: Load embedding model
print("\n[Test 3] Loading embedding model (downloads on first run)...")
from src.rag.embeddings import build_embeddings

embeddings = build_embeddings("all-MiniLM-L6-v2")
print("  => Embedding model ready")

# Test 4: Build FAISS index
print("\n[Test 4] Building FAISS vector index...")
from src.rag.vectorstore import build_vector_store

store_dir = Path("vector_store")
vs = build_vector_store(chunks, embeddings, store_dir)
print("  => FAISS index built")

# Test 5: Semantic retrieval
print("\n[Test 5] Testing semantic retrieval...")
from src.rag.vectorstore import retrieve_documents

query = "bearing failure vibration analysis rotating equipment maintenance"
results = retrieve_documents(vs, query, top_k=3)
assert len(results) > 0, "No results returned!"
print(f"  => Query: '{query}'")
print(f"  => Retrieved {len(results)} chunk(s):")
for i, r in enumerate(results, 1):
    src = r.metadata.get("source", "unknown")
    preview = r.page_content[:90].strip().replace("\n", " ")
    print(f"     [{i}] {src}: {preview}...")

# Test 6: Persistence - save and reload
print("\n[Test 6] Verifying FAISS persistence (save -> load -> query)...")
from src.rag.vectorstore import load_vector_store

index_file = store_dir / "index.faiss"
assert index_file.exists(), f"FAISS index file not found at {index_file}"
print(f"  => index.faiss exists ({index_file.stat().st_size:,} bytes)")

vs_reloaded = load_vector_store(store_dir, embeddings)
results2 = retrieve_documents(vs_reloaded, query, top_k=2)
assert len(results2) > 0, "Reloaded index returned no results!"
print(f"  => Reloaded index returned {len(results2)} result(s) - persistence OK")

# Test 7: Text cleaner
print("\n[Test 7] Testing text cleaner utility...")
from src.utils.text_cleaner import clean_text

raw = "  Hello   World  \n\n\n\nPage 1\n\nSome technical text.\n\n\n"
cleaned = clean_text(raw)
assert "Page 1" not in cleaned, "Page number not removed!"
assert "   " not in cleaned, "Extra spaces not collapsed!"
print(f"  => Input:   {raw!r}")
print(f"  => Cleaned: {cleaned!r}")

print("\n" + "=" * 55)
print("  ALL RAG TESTS PASSED")
print("=" * 55)
