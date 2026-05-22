from pathlib import Path
from datetime import datetime
import os
import re
from dotenv import load_dotenv
import ollama

# Load .env from the repository root (if present) so environment variables
# like OBSIDIAN_VAULT are available to the script without manual setup.
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()

MODEL = "llama3.2:3b"


def get_vault_path() -> Path:
    vault = os.environ.get("OBSIDIAN_VAULT")

    if not vault:
        raise EnvironmentError(
            "OBSIDIAN_VAULT is not set. Set it in PowerShell like this:\n"
            '$env:OBSIDIAN_VAULT="C:\\Users\\YourName\\Documents\\Obsidian\\AI-Second-Brain"'
        )

    path = Path(vault)

    if not path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {path}")

    return path


def safe_filename(title: str) -> str:
    title = re.sub(r'[<>:"/\\\\|?*]', "", title)
    title = title.strip()
    return title[:80] if title else "Untitled Note"


def generate_note(topic: str) -> str:
    prompt = f"""
Create a clean Obsidian Markdown note on this topic:

{topic}

Use this exact structure:

# {topic}

## Simple Explanation

## Why It Matters

## Key Concepts

## Real-World Example

## Beginner Project Idea

## Advanced Project Idea

## Quick Quiz

## Related Notes

Use Obsidian backlinks in the Related Notes section using [[...]] format.

Keep the note beginner-friendly, practical, and useful for someone learning AI agents, MCP, LangGraph, Claude Code, ADK, and Antigravity.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI tutor and Obsidian second-brain note maker.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


def save_note(topic: str, content: str) -> Path:
    vault = get_vault_path()

    inbox_folder = vault / "00-Inbox"
    inbox_folder.mkdir(exist_ok=True)

    filename = safe_filename(topic) + ".md"
    note_path = inbox_folder / filename

    frontmatter = f"""---
created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
source: AI Note Assistant
status: draft
---

"""

    note_path.write_text(frontmatter + content, encoding="utf-8")

    return note_path


def main():
    print("AI Note Assistant for Obsidian")
    print("--------------------------------")

    topic = input("Enter topic for your Obsidian note: ").strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    print("\nGenerating note using Ollama...")
    content = generate_note(topic)

    print("\nSaving note to Obsidian...")
    note_path = save_note(topic, content)

    print("\nDone!")
    print(f"Saved note at: {note_path}")
    print("\nOpen Obsidian → 00-Inbox to see your generated note.")


if __name__ == "__main__":
    main()