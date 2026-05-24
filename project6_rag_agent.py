from pathlib import Path
from typing import TypedDict, List, Dict, Any
from datetime import datetime
import os
import re

from dotenv import load_dotenv
import chromadb
import ollama
from langgraph.graph import StateGraph, START, END


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


class RAGAgentState(TypedDict):
    original_question: str
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    context_text: str
    answer: str
    grounding_report: str
    final_note: str
    saved_path: str


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


def rewrite_query_agent(state: RAGAgentState) -> RAGAgentState:
    print("Query Rewriter Agent: Rewriting question for retrieval...")

    question = state["original_question"]

    prompt = f"""
Rewrite the user's question into a better semantic search query for retrieving relevant Obsidian notes.

Original question:
{question}

Rules:
- Keep the rewritten query short.
- Include important technical terms.
- Do not answer the question.
- Return only the rewritten query.
"""

    rewritten = ask_ollama(
        prompt,
        "You are a query rewriting agent for vector database retrieval.",
    )

    state["rewritten_query"] = rewritten.strip().replace('"', "")
    print(f"Rewritten query: {state['rewritten_query']}")

    return state


def retrieve_context_agent(state: RAGAgentState) -> RAGAgentState:
    print("Retriever Agent: Searching ChromaDB vector database...")

    collection = get_collection()
    count = collection.count()

    if count == 0:
        raise RuntimeError(
            "Vector database is empty. Run this first:\n"
            "python project5_vector_search_obsidian.py index"
        )

    query_embedding = embed_text(state["rewritten_query"])

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_chunks = []

    for doc, meta, distance in zip(docs, metas, distances):
        retrieved_chunks.append(
            {
                "source_path": meta.get("source_path", "unknown"),
                "title": meta.get("title", "unknown"),
                "chunk_index": meta.get("chunk_index", "unknown"),
                "distance": distance,
                "content": doc,
            }
        )

    state["retrieved_chunks"] = retrieved_chunks

    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"""## Source {i}
Path: {chunk["source_path"]}
Title: {chunk["title"]}
Chunk: {chunk["chunk_index"]}
Distance: {chunk["distance"]}

Content:
{chunk["content"]}
"""
        )

    state["context_text"] = "\n\n---\n\n".join(context_blocks)

    print(f"Retrieved {len(retrieved_chunks)} chunks.")

    return state


def answer_generator_agent(state: RAGAgentState) -> RAGAgentState:
    print("Answer Generator Agent: Creating grounded answer...")

    prompt = f"""
You are a RAG answer generation agent.

Answer the user's question using ONLY the retrieved Obsidian context.

User question:
{state["original_question"]}

Rewritten retrieval query:
{state["rewritten_query"]}

Retrieved context:
{state["context_text"]}

Important rules:
- Do not use outside knowledge unless clearly marked as general background.
- Prefer information from the retrieved notes.
- Mention the source note paths used.
- If context is insufficient, clearly say what is missing.

Use this format:

## Answer

## Explanation

## Key Points

## Source Notes Used

## What Is Missing / What To Learn Next
"""

    answer = ask_ollama(
        prompt,
        "You are a careful RAG agent. You answer only from provided context and cite source note paths.",
    )

    state["answer"] = answer

    return state


def grounding_checker_agent(state: RAGAgentState) -> RAGAgentState:
    print("Grounding Checker Agent: Checking answer quality...")

    prompt = f"""
You are a grounding checker agent.

Check whether the answer is properly grounded in the retrieved context.

Original question:
{state["original_question"]}

Retrieved context:
{state["context_text"]}

Generated answer:
{state["answer"]}

Give:

## Grounding Check

- Is the answer supported by the retrieved context?
- Are source notes mentioned?
- Are there unsupported claims?
- Is the answer useful?

## Risk Level

Choose one:
Low / Medium / High

## Fix Suggestions

Give practical improvements.
"""

    report = ask_ollama(
        prompt,
        "You are a strict grounding and quality checker for RAG systems.",
    )

    state["grounding_report"] = report

    return state


def assemble_note_agent(state: RAGAgentState) -> RAGAgentState:
    print("Save Agent: Assembling final Markdown note...")

    question = state["original_question"]

    sources = []
    for chunk in state["retrieved_chunks"]:
        source_path = chunk["source_path"]
        if source_path not in sources:
            sources.append(source_path)

    source_list = "\n".join([f"- `{src}`" for src in sources])

    state["final_note"] = f"""---
created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
source: LangGraph RAG Agent
status: draft
tags:
  - ai
  - rag
  - vector-database
  - langgraph
  - obsidian
---

# {question}

#ai #rag #vector-database #langgraph #obsidian

## RAG Agent Workflow

Original Question  
↓  
Query Rewriter Agent  
↓  
Retriever Agent  
↓  
Answer Generator Agent  
↓  
Grounding Checker Agent  
↓  
Save to Obsidian Agent  

---

## Original Question

{state["original_question"]}

---

## Rewritten Retrieval Query

{state["rewritten_query"]}

---

## Final Answer

{state["answer"]}

---

## Grounding Report

{state["grounding_report"]}

---

## Source Notes Used

{source_list}

---

## Retrieved Context

{state["context_text"]}

---

## Related Notes

- [[AI Agents]]
- [[LangGraph]]
- [[MCP]]
- [[Vector Databases for RAG]]
- [[Project 5 - Vector Search over Obsidian]]
- [[Project 6 - RAG Agent over Obsidian]]
"""

    return state


def save_to_obsidian_agent(state: RAGAgentState) -> RAGAgentState:
    print("Save Agent: Saving final note to Obsidian...")

    vault = get_vault_path()

    target_folder = vault / "06-Advanced-Agent-Engineering" / "RAG Agent Runs"
    target_folder.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(state["original_question"]) + " - RAG Agent.md"
    note_path = target_folder / filename

    if note_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_path = target_folder / f"{safe_filename(state['original_question'])} - RAG Agent {timestamp}.md"

    note_path.write_text(state["final_note"], encoding="utf-8")

    state["saved_path"] = str(note_path)

    return state


def build_graph():
    graph = StateGraph(RAGAgentState)

    graph.add_node("rewrite_query_agent", rewrite_query_agent)
    graph.add_node("retrieve_context_agent", retrieve_context_agent)
    graph.add_node("answer_generator_agent", answer_generator_agent)
    graph.add_node("grounding_checker_agent", grounding_checker_agent)
    graph.add_node("assemble_note_agent", assemble_note_agent)
    graph.add_node("save_to_obsidian_agent", save_to_obsidian_agent)

    graph.add_edge(START, "rewrite_query_agent")
    graph.add_edge("rewrite_query_agent", "retrieve_context_agent")
    graph.add_edge("retrieve_context_agent", "answer_generator_agent")
    graph.add_edge("answer_generator_agent", "grounding_checker_agent")
    graph.add_edge("grounding_checker_agent", "assemble_note_agent")
    graph.add_edge("assemble_note_agent", "save_to_obsidian_agent")
    graph.add_edge("save_to_obsidian_agent", END)

    return graph.compile()


def main():
    print("LangGraph RAG Agent over Obsidian")
    print("---------------------------------")

    question = input("Enter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    initial_state: RAGAgentState = {
        "original_question": question,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "context_text": "",
        "answer": "",
        "grounding_report": "",
        "final_note": "",
        "saved_path": "",
    }

    app = build_graph()
    result = app.invoke(initial_state)

    print("\nDone!")
    print(f"Saved note at: {result['saved_path']}")
    print("\nOpen Obsidian → 06-Advanced-Agent-Engineering → RAG Agent Runs")


if __name__ == "__main__":
    main()