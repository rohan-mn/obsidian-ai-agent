from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
import chromadb
import ollama


# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()


CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_obsidian_db"
COLLECTION_NAME = "obsidian_second_brain"


def get_vault_path() -> Path:
    vault = os.environ.get("OBSIDIAN_VAULT")

    if not vault:
        raise EnvironmentError(
            "OBSIDIAN_VAULT is not set. Add this to .env:\n"
            "OBSIDIAN_VAULT=C:\\Path\\To\\AI-Second-Brain"
        )

    path = Path(vault)

    if not path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {path}")

    return path


def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(reset: bool = False):
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_markdown_files() -> List[Dict[str, Any]]:
    vault = get_vault_path()
    notes = []

    for path in vault.rglob("*.md"):
        # Skip internal Obsidian config if present
        if ".obsidian" in path.parts:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        text = clean_text(text)

        if not text:
            continue

        relative_path = str(path.relative_to(vault))

        notes.append(
            {
                "path": path,
                "relative_path": relative_path,
                "title": path.stem,
                "text": text,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )

    return notes


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking.
    Good enough for today's vector database practice.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

    return chunks


def make_chunk_id(relative_path: str, chunk_index: int, chunk_text_value: str) -> str:
    raw = f"{relative_path}:{chunk_index}:{chunk_text_value[:100]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def embed_text(text: str) -> List[float]:
    """
    Uses Ollama local embeddings.
    Supports both newer ollama.embed() and older ollama.embeddings().
    """
    try:
        response = ollama.embed(
            model=EMBED_MODEL,
            input=text,
        )

        if "embeddings" in response:
            return response["embeddings"][0]

    except Exception:
        pass

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text,
    )

    return response["embedding"]


def index_obsidian_notes():
    print("Building fresh Chroma vector index...")
    print("------------------------------------")

    notes = read_markdown_files()
    print(f"Found {len(notes)} Markdown notes.")

    collection = get_collection(reset=True)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    total_chunks = 0

    for note in notes:
        chunks = chunk_text(note["text"])

        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(note["relative_path"], i, chunk)

            metadata = {
                "source_path": note["relative_path"],
                "title": note["title"],
                "chunk_index": i,
                "modified_at": note["modified_at"],
            }

            print(f"Embedding: {note['relative_path']} | chunk {i + 1}/{len(chunks)}")

            vector = embed_text(chunk)

            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(vector)
            metadatas.append(metadata)

            total_chunks += 1

            # Batch upload every 50 chunks
            if len(ids) >= 50:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                ids, documents, embeddings, metadatas = [], [], [], []

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print("\nIndexing complete.")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Chroma DB saved at: {CHROMA_DIR}")


def search_vector_db(query: str, top_k: int = 5):
    collection = get_collection(reset=False)

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


def print_search_results(query: str, top_k: int = 5):
    print(f"\nSemantic search for: {query}")
    print("--------------------------------")

    results = search_vector_db(query, top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        print("No results found. Run indexing first:")
        print("python project5_vector_search_obsidian.py index")
        return

    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        print(f"\nResult {i}")
        print(f"Source: {meta['source_path']}")
        print(f"Title: {meta['title']}")
        print(f"Chunk: {meta['chunk_index']}")
        print(f"Distance: {distance}")
        print("Snippet:")
        print(doc[:700].replace("\n", " "))
        print("-" * 80)


def ask_question(query: str, top_k: int = 5):
    results = search_vector_db(query, top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        print("No relevant chunks found. Run indexing first.")
        return

    context_blocks = []

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        context_blocks.append(
            f"Source {i}: {meta['source_path']}\n"
            f"Title: {meta['title']}\n"
            f"Content:\n{doc}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are an AI second-brain assistant.

Answer the user's question using ONLY the retrieved Obsidian note context below.
If the context is insufficient, say what is missing.
Cite the source note paths in the answer.

User question:
{query}

Retrieved context:
{context}

Answer format:

## Answer

## Key Points

## Source Notes

## Suggested Next Note to Create
"""

    print("\nGenerating answer using retrieved Obsidian context...\n")

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful RAG assistant that answers from provided notes only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    print(response["message"]["content"])


def save_rag_answer(query: str, top_k: int = 5):
    results = search_vector_db(query, top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        print("No relevant chunks found. Run indexing first.")
        return

    context_blocks = []

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        context_blocks.append(
            f"## Source {i}: {meta['source_path']}\n\n"
            f"{doc}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
Create an Obsidian Markdown note answering this question using the retrieved context.

Question:
{query}

Retrieved context:
{context}

Use this structure:

# {query}

## Answer

## Explanation

## Important Points

## Related Notes

## Source Notes

Use Obsidian backlinks wherever useful.
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Obsidian note-maker and RAG assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    vault = get_vault_path()
    target_folder = vault / "06-Advanced-Agent-Engineering" / "Vector Search Runs"
    target_folder.mkdir(parents=True, exist_ok=True)

    safe_query = re.sub(r'[<>:"/\\\\|?*]', "", query).strip()[:80]
    note_path = target_folder / f"{safe_query} - RAG Answer.md"

    final_note = f"""---
created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
source: Vector Search RAG Agent
status: draft
tags:
  - ai
  - rag
  - vector-database
  - obsidian
---

{response["message"]["content"]}

---

# Retrieved Context Used

{context}
"""

    note_path.write_text(final_note, encoding="utf-8")

    print(f"Saved RAG answer to: {note_path}")


def show_stats():
    collection = get_collection(reset=False)

    try:
        count = collection.count()
    except Exception:
        count = 0

    print("Vector Database Stats")
    print("---------------------")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total chunks: {count}")
    print(f"Database path: {CHROMA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Vector Search over Obsidian using ChromaDB + Ollama")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("index", help="Index all Obsidian Markdown notes")

    search_parser = subparsers.add_parser("search", help="Semantic search over notes")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--top-k", type=int, default=5)

    ask_parser = subparsers.add_parser("ask", help="Ask a question using retrieved note context")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--top-k", type=int, default=5)

    save_parser = subparsers.add_parser("save-answer", help="Ask and save RAG answer to Obsidian")
    save_parser.add_argument("query", type=str)
    save_parser.add_argument("--top-k", type=int, default=5)

    subparsers.add_parser("stats", help="Show vector DB stats")

    args = parser.parse_args()

    if args.command == "index":
        index_obsidian_notes()

    elif args.command == "search":
        print_search_results(args.query, args.top_k)

    elif args.command == "ask":
        ask_question(args.query, args.top_k)

    elif args.command == "save-answer":
        save_rag_answer(args.query, args.top_k)

    elif args.command == "stats":
        show_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()