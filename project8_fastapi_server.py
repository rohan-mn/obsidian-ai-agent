from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import chromadb
import ollama


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Obsidian AI Agent API",
    description="FastAPI wrapper for Obsidian, Ollama, ChromaDB, and RAG agent workflows.",
    version="1.0.0",
)


# ---------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    obsidian_vault_found: bool
    chroma_collection: str
    chroma_chunks: int
    chat_model: str
    embed_model: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language query for semantic search")
    top_k: int = Field(5, ge=1, le=20)


class SearchResult(BaseModel):
    rank: int
    source_path: str
    title: str
    chunk_index: int
    distance: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    results: List[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(..., description="Question to answer using Obsidian vector search")
    top_k: int = Field(6, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks: List[SearchResult]


class AgentRunRequest(BaseModel):
    question: str = Field(..., description="Question for full RAG agent run")
    top_k: int = Field(6, ge=1, le=20)
    save_to_obsidian: bool = True


class AgentRunResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    grounding_report: str
    sources: List[str]
    saved_path: Optional[str]
    retrieved_chunks: List[SearchResult]

def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> bool:
    expected_key = os.environ.get("AGENT_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="Server API key is not configured. Set AGENT_API_KEY in .env."
        )

    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return True


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_vault_path() -> Path:
    vault = os.environ.get("OBSIDIAN_VAULT")

    if not vault:
        raise RuntimeError(
            "OBSIDIAN_VAULT is not set. Add it to .env."
        )

    path = Path(vault)

    if not path.exists():
        raise RuntimeError(f"Vault path does not exist: {path}")

    return path


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def safe_filename(title: str) -> str:
    title = re.sub(r'[<>:"/\\\\|?*]', "", title)
    title = title.strip()
    return title[:80] if title else "Untitled"


def ask_ollama(prompt: str, system_message: str = "") -> str:
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_message or "You are a helpful AI assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


def embed_text(text: str) -> List[float]:
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


def rewrite_query(question: str) -> str:
    prompt = f"""
Rewrite this user question into a short semantic search query for retrieving relevant Obsidian notes.

Original question:
{question}

Rules:
- Keep it short.
- Include important technical terms.
- Do not answer.
- Return only the rewritten query.
"""

    rewritten = ask_ollama(
        prompt,
        "You are a query rewriting agent for vector database retrieval.",
    )

    return rewritten.strip().replace('"', "")


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "Vector database is empty. Run: python project5_vector_search_obsidian.py index"
        )

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []

    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        chunks.append(
            {
                "rank": i,
                "source_path": meta.get("source_path", "unknown"),
                "title": meta.get("title", "unknown"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "distance": float(distance),
                "content": doc,
                "snippet": doc[:700].replace("\n", " "),
            }
        )

    return chunks


def chunks_to_search_results(chunks: List[Dict[str, Any]]) -> List[SearchResult]:
    return [
        SearchResult(
            rank=chunk["rank"],
            source_path=chunk["source_path"],
            title=chunk["title"],
            chunk_index=chunk["chunk_index"],
            distance=chunk["distance"],
            snippet=chunk["snippet"],
        )
        for chunk in chunks
    ]


def build_context(chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for chunk in chunks:
        context_blocks.append(
            f"""## Source {chunk["rank"]}
Path: {chunk["source_path"]}
Title: {chunk["title"]}
Chunk: {chunk["chunk_index"]}
Distance: {chunk["distance"]}

Content:
{chunk["content"]}
"""
        )

    return "\n\n---\n\n".join(context_blocks)


def generate_rag_answer(question: str, rewritten_query: str, chunks: List[Dict[str, Any]]) -> str:
    context = build_context(chunks)

    prompt = f"""
You are a RAG answer generation agent.

Answer the user's question using ONLY the retrieved Obsidian context.

User question:
{question}

Rewritten retrieval query:
{rewritten_query}

Retrieved context:
{context}

Rules:
- Prefer information from retrieved notes.
- Mention source note paths.
- If context is insufficient, say what is missing.
- Do not invent unsupported details.

Use this format:

## Answer

## Explanation

## Key Points

## Source Notes Used

## What Is Missing / What To Learn Next
"""

    return ask_ollama(
        prompt,
        "You are a careful RAG agent. Answer only from provided context and cite source note paths.",
    )


def check_grounding(question: str, answer: str, chunks: List[Dict[str, Any]]) -> str:
    context = build_context(chunks)

    prompt = f"""
You are a grounding checker agent.

Check whether the generated answer is supported by the retrieved context.

Question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}

Give:

## Grounding Check
- Is the answer supported?
- Are source notes mentioned?
- Are there unsupported claims?
- Is it useful?

## Risk Level
Choose one: Low / Medium / High

## Fix Suggestions
Give practical improvements.
"""

    return ask_ollama(
        prompt,
        "You are a strict grounding and quality checker for RAG systems.",
    )


def save_agent_run_to_obsidian(
    question: str,
    rewritten_query: str,
    answer: str,
    grounding_report: str,
    chunks: List[Dict[str, Any]],
) -> str:
    vault = get_vault_path()

    target_folder = vault / "06-Advanced-Agent-Engineering" / "FastAPI Runs"
    target_folder.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(question) + " - FastAPI Agent.md"
    note_path = target_folder / filename

    if note_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_path = target_folder / f"{safe_filename(question)} - FastAPI Agent {timestamp}.md"

    sources = []
    for chunk in chunks:
        if chunk["source_path"] not in sources:
            sources.append(chunk["source_path"])

    source_list = "\n".join([f"- `{src}`" for src in sources])
    context = build_context(chunks)

    final_note = f"""---
created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
source: FastAPI RAG Agent
status: draft
tags:
  - ai
  - rag
  - fastapi
  - obsidian
  - vector-database
---

# {question}

#ai #rag #fastapi #obsidian #vector-database

## API Workflow

HTTP Request  
↓  
FastAPI Endpoint  
↓  
Query Rewriter  
↓  
ChromaDB Retriever  
↓  
Ollama Answer Generator  
↓  
Grounding Checker  
↓  
Save to Obsidian  

---

## Original Question

{question}

---

## Rewritten Query

{rewritten_query}

---

## Answer

{answer}

---

## Grounding Report

{grounding_report}

---

## Source Notes Used

{source_list}

---

## Retrieved Context

{context}

---

## Related Notes

- [[AI Agents]]
- [[LangGraph]]
- [[MCP]]
- [[Vector Databases for RAG]]
- [[Production Deployment]]
- [[Project 8 - FastAPI Production Wrapper]]
"""

    note_path.write_text(final_note, encoding="utf-8")

    return str(note_path)


# ---------------------------------------------------------
# API Routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Obsidian AI Agent API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health(_: bool = Depends(verify_api_key)):
    try:
        vault_found = get_vault_path().exists()
    except Exception:
        vault_found = False

    try:
        collection = get_collection()
        chunk_count = collection.count()
    except Exception:
        chunk_count = 0

    return HealthResponse(
        status="ok",
        obsidian_vault_found=vault_found,
        chroma_collection=COLLECTION_NAME,
        chroma_chunks=chunk_count,
        chat_model=CHAT_MODEL,
        embed_model=EMBED_MODEL,
    )


@app.get("/stats")
def stats(_: bool = Depends(verify_api_key)):
    try:
        collection = get_collection()
        return {
            "collection": COLLECTION_NAME,
            "total_chunks": collection.count(),
            "database_path": str(CHROMA_DIR),
            "vault_path": str(get_vault_path()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_notes_api(
    request: SearchRequest,
    _: bool = Depends(verify_api_key),
):
    try:
        chunks = await run_in_threadpool(vector_search, request.query, request.top_k)
        return SearchResponse(
            query=request.query,
            results=chunks_to_search_results(chunks),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_api(
    request: AskRequest,
    _: bool = Depends(verify_api_key),
):
    try:
        rewritten_query = await run_in_threadpool(rewrite_query, request.question)
        chunks = await run_in_threadpool(vector_search, rewritten_query, request.top_k)
        answer = await run_in_threadpool(
            generate_rag_answer,
            request.question,
            rewritten_query,
            chunks,
        )

        sources = []
        for chunk in chunks:
            if chunk["source_path"] not in sources:
                sources.append(chunk["source_path"])

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks_to_search_results(chunks),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/run", response_model=AgentRunResponse)
async def agent_run_api(
    request: AgentRunRequest,
    _: bool = Depends(verify_api_key),
):
    try:
        rewritten_query = await run_in_threadpool(rewrite_query, request.question)
        chunks = await run_in_threadpool(vector_search, rewritten_query, request.top_k)
        answer = await run_in_threadpool(
            generate_rag_answer,
            request.question,
            rewritten_query,
            chunks,
        )
        grounding_report = await run_in_threadpool(
            check_grounding,
            request.question,
            answer,
            chunks,
        )

        saved_path = None

        if request.save_to_obsidian:
            saved_path = await run_in_threadpool(
                save_agent_run_to_obsidian,
                request.question,
                rewritten_query,
                answer,
                grounding_report,
                chunks,
            )

        sources = []
        for chunk in chunks:
            if chunk["source_path"] not in sources:
                sources.append(chunk["source_path"])

        return AgentRunResponse(
            question=request.question,
            rewritten_query=rewritten_query,
            answer=answer,
            grounding_report=grounding_report,
            sources=sources,
            saved_path=saved_path,
            retrieved_chunks=chunks_to_search_results(chunks),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Local run command
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "project8_fastapi_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )