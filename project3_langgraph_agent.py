from pathlib import Path
from typing import TypedDict
from datetime import datetime
import os
import re

from dotenv import load_dotenv
import ollama
from langgraph.graph import StateGraph, START, END


# Load .env file if available
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()


MODEL = "llama3.2:3b"


class AgentState(TypedDict):
    topic: str
    explanation: str
    quiz: str
    project_ideas: str
    final_note: str
    saved_path: str


def get_vault_path() -> Path:
    vault = os.environ.get("OBSIDIAN_VAULT")

    if not vault:
        raise EnvironmentError(
            "OBSIDIAN_VAULT is not set. Add it to .env like this:\n"
            "OBSIDIAN_VAULT=C:\\Path\\To\\AI-Second-Brain"
        )

    path = Path(vault)

    if not path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {path}")

    return path


def safe_filename(title: str) -> str:
    title = re.sub(r'[<>:"/\\\\|?*]', "", title)
    title = title.strip()
    return title[:80] if title else "Untitled"


def ask_ollama(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI tutor, research assistant, and Obsidian note maker. "
                    "Explain clearly, practically, and in Markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


def generate_explanation(state: AgentState) -> AgentState:
    print("Step 1: Generating explanation...")

    topic = state["topic"]

    prompt = f"""
Create a beginner-friendly but useful explanation of this topic:

{topic}

Include:

## Simple Meaning

## Why It Matters

## Key Concepts

## Real-World Example

## Common Mistakes

## Related Obsidian Notes

Use backlinks like:
[[AI Agents]]
[[MCP]]
[[LangGraph]]
[[Claude Code]]
[[Google ADK]]
[[Antigravity]]
"""

    state["explanation"] = ask_ollama(prompt)
    return state


def generate_quiz(state: AgentState) -> AgentState:
    print("Step 2: Generating quiz...")

    topic = state["topic"]

    prompt = f"""
Create a quiz for this topic:

{topic}

Include:

## Multiple Choice Questions

Create 5 MCQs with 4 options each.

## Short Answer Questions

Create 5 short answer questions.

## Answer Key

Give clear answers.
"""

    state["quiz"] = ask_ollama(prompt)
    return state


def generate_project_ideas(state: AgentState) -> AgentState:
    print("Step 3: Generating project ideas...")

    topic = state["topic"]

    prompt = f"""
Suggest practical projects for learning this topic:

{topic}

Include:

## Beginner Project

- Goal
- Tools needed
- Steps
- Expected output

## Intermediate Project

- Goal
- Tools needed
- Steps
- Expected output

## Advanced Project

- Goal
- Tools needed
- Steps
- Expected output

Make the projects relevant to Obsidian, Markdown, AI agents, MCP, LangGraph, ADK, or agentic coding.
"""

    state["project_ideas"] = ask_ollama(prompt)
    return state


def assemble_note(state: AgentState) -> AgentState:
    print("Step 4: Assembling final Markdown note...")

    topic = state["topic"]

    state["final_note"] = f"""---
created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
source: LangGraph Research Agent
status: draft
tags:
  - ai
  - langgraph
  - second-brain
---

# {topic}

#ai #langgraph #second-brain

## Agent Workflow

This note was generated using a LangGraph workflow:

Topic Input  
↓  
Generate Explanation  
↓  
Generate Quiz  
↓  
Generate Project Ideas  
↓  
Assemble Markdown  
↓  
Save to Obsidian  

---

## Explanation

{state["explanation"]}

---

## Quiz

{state["quiz"]}

---

## Project Ideas

{state["project_ideas"]}

---

## Related Notes

- [[AI Agents]]
- [[MCP]]
- [[LangGraph]]
- [[Claude Code]]
- [[Google ADK]]
- [[Antigravity]]
- [[Project 1 - AI Note Assistant]]
- [[Project 3 - LangGraph Research Agent]]
"""

    return state


def save_to_obsidian(state: AgentState) -> AgentState:
    print("Step 5: Saving note to Obsidian...")

    vault = get_vault_path()

    target_folder = vault / "01-Learning-Roadmap"
    target_folder.mkdir(exist_ok=True)

    filename = safe_filename(state["topic"]) + " - Research Agent.md"
    note_path = target_folder / filename

    if note_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = safe_filename(state["topic"]) + f" - Research Agent {timestamp}.md"
        note_path = target_folder / filename

    note_path.write_text(state["final_note"], encoding="utf-8")

    state["saved_path"] = str(note_path)

    return state


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("generate_quiz", generate_quiz)
    graph.add_node("generate_project_ideas", generate_project_ideas)
    graph.add_node("assemble_note", assemble_note)
    graph.add_node("save_to_obsidian", save_to_obsidian)

    graph.add_edge(START, "generate_explanation")
    graph.add_edge("generate_explanation", "generate_quiz")
    graph.add_edge("generate_quiz", "generate_project_ideas")
    graph.add_edge("generate_project_ideas", "assemble_note")
    graph.add_edge("assemble_note", "save_to_obsidian")
    graph.add_edge("save_to_obsidian", END)

    return graph.compile()


def main():
    print("LangGraph Research Agent for Obsidian")
    print("-------------------------------------")

    topic = input("Enter research topic: ").strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    app = build_graph()

    initial_state: AgentState = {
        "topic": topic,
        "explanation": "",
        "quiz": "",
        "project_ideas": "",
        "final_note": "",
        "saved_path": "",
    }

    result = app.invoke(initial_state)

    print("\nDone!")
    print(f"Saved note at: {result['saved_path']}")
    print("\nOpen Obsidian → 01-Learning-Roadmap to see the generated note.")


if __name__ == "__main__":
    main()