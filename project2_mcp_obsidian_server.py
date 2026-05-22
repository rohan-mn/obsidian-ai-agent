from pathlib import Path
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()


mcp = FastMCP("obsidian-second-brain")


def get_vault_path() -> Path:
    vault = os.environ.get("OBSIDIAN_VAULT")

    if not vault:
        raise EnvironmentError(
            "OBSIDIAN_VAULT is not set. Add this to .env:\n"
            "OBSIDIAN_VAULT=C:\\Path\\To\\AI-Second-Brain"
        )

    path = Path(vault).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {path}")

    return path


def safe_vault_path(relative_path: str) -> Path:
    """
    Prevents the tool from reading/writing outside the Obsidian vault.
    """
    vault = get_vault_path()
    target = (vault / relative_path).resolve()

    if not str(target).startswith(str(vault)):
        raise PermissionError("Access outside the Obsidian vault is not allowed.")

    return target


@mcp.tool()
def list_notes(folder: str = "") -> list[str]:
    """
    List Markdown notes inside the Obsidian vault.
    Example folder: 02-Concepts
    """
    vault = get_vault_path()

    if folder.strip():
        target = safe_vault_path(folder)
    else:
        target = vault

    if not target.exists():
        return [f"Folder does not exist: {folder}"]

    notes = []

    for path in target.rglob("*.md"):
        notes.append(str(path.relative_to(vault)))

    return sorted(notes)[:200]


@mcp.tool()
def read_note(note_path: str) -> str:
    """
    Read a Markdown note from the Obsidian vault.
    Example: 02-Concepts/MCP.md
    """
    path = safe_vault_path(note_path)

    if not path.exists():
        return f"Note not found: {note_path}"

    if path.suffix.lower() != ".md":
        return "Only Markdown .md files can be read."

    return path.read_text(encoding="utf-8", errors="ignore")


@mcp.tool()
def search_notes(query: str) -> list[str]:
    """
    Search Markdown notes by keyword.
    """
    vault = get_vault_path()
    query_lower = query.lower().strip()

    if not query_lower:
        return ["Search query cannot be empty."]

    matches = []

    for path in vault.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")

        if query_lower in text.lower() or query_lower in path.name.lower():
            matches.append(str(path.relative_to(vault)))

    return sorted(matches)[:100]


@mcp.tool()
def create_note(note_path: str, content: str) -> str:
    """
    Create a new Markdown note in the Obsidian vault.
    Example note_path: 00-Inbox/New Idea.md
    """
    if not note_path.endswith(".md"):
        note_path = note_path + ".md"

    path = safe_vault_path(note_path)

    if path.exists():
        return f"Note already exists: {note_path}"

    path.parent.mkdir(parents=True, exist_ok=True)

    frontmatter = f"""---
created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
source: MCP Obsidian Server
status: draft
---

"""

    path.write_text(frontmatter + content, encoding="utf-8")

    return f"Created note: {note_path}"


@mcp.tool()
def append_to_note(note_path: str, content: str) -> str:
    """
    Append content to an existing Markdown note.
    Example: append_to_note("02-Concepts/MCP.md", "New text")
    """
    path = safe_vault_path(note_path)

    if not path.exists():
        return f"Note not found: {note_path}"

    if path.suffix.lower() != ".md":
        return "Only Markdown .md files can be edited."

    old_content = path.read_text(encoding="utf-8", errors="ignore")
    new_content = old_content + "\n\n" + content

    path.write_text(new_content, encoding="utf-8")

    return f"Appended content to: {note_path}"


def self_test():
    """
    Simple local test without an MCP client.
    """
    print("Running MCP Obsidian Server self-test...")
    print("--------------------------------------")

    vault = get_vault_path()
    print(f"Vault found: {vault}")

    print("\nListing notes:")
    notes = list_notes()
    for note in notes[:10]:
        print("-", note)

    print("\nSearching for MCP:")
    results = search_notes("MCP")
    for result in results[:10]:
        print("-", result)

    print("\nSelf-test complete.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        mcp.run()