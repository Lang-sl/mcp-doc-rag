#!/usr/bin/env python3
"""Interactive configuration setup for RAG-Engine.

Run this script once after cloning the repository:

    python setup_config.py

It will:
  1. Create config.yaml from the template
  2. Help you configure document sources
  3. Verify Ollama is available (optional)
  4. Print next steps
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).resolve().parent / "config.example.yaml"
TARGET = Path(__file__).resolve().parent / "config.yaml"


def header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def ask(prompt: str, default: str = "") -> str:
    """Ask the user for input with an optional default."""
    if default:
        result = input(f"  {prompt} [{default}]: ").strip()
    else:
        result = input(f"  {prompt}: ").strip()
    return result if result else default


def check_ollama() -> bool:
    """Check if Ollama is running and has the embedding model."""
    try:
        import ollama

        client = ollama.Client(host="http://localhost:11434")
        resp = client.list()
        models = [m.get("model", "") for m in resp.get("models", [])]
        if any("nomic-embed-text" in m for m in models):
            print("  [OK] Ollama is running, nomic-embed-text is available")
            return True
        else:
            print("  [WARN] Ollama is running but nomic-embed-text is not pulled")
            pull = ask("Pull it now? (y/n)", "y")
            if pull.lower() == "y":
                print("  Pulling nomic-embed-text (~274MB)...")
                os.system("ollama pull nomic-embed-text")
                return True
            return False
    except Exception:
        print("  [WARN] Ollama is not running. Start it with: ollama serve")
        return False


def main() -> int:
    header("RAG-Engine Configuration Setup")

    # --- Step 1: Create config.yaml ---
    if TARGET.exists():
        print(f"\n  config.yaml already exists at: {TARGET}")
        overwrite = ask("Overwrite? (y/n)", "n")
        if overwrite.lower() != "y":
            print("  Keeping existing config.yaml")
        else:
            shutil.copy(TEMPLATE, TARGET)
            print(f"  config.yaml overwritten from template")
    else:
        shutil.copy(TEMPLATE, TARGET)
        print(f"\n  Created: {TARGET}")

    # --- Step 2: Load and edit ---
    with open(TARGET, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    header("Document Sources")
    print("\n  Add paths to your SDK documentation (Doxygen HTML, PDFs, or C++ headers).")
    print("  Leave the label empty and press Enter to finish.\n")

    sources = config.get("doc_sources", {})
    # Remove the example placeholder
    example_key = "example_sdk"
    if example_key in sources and sources[example_key] == "/path/to/your/sdk/docs":
        del sources[example_key]

    while True:
        label = ask("Source label (e.g., my_sdk)")
        if not label:
            break

        path = ask(f"  Path for '{label}'")
        if not path:
            print("  Skipped — no path given")
            continue

        if not os.path.isdir(path):
            print(f"  [WARN] Directory not found: {path}")
            still_add = ask("  Add anyway? (y/n)", "n")
            if still_add.lower() != "y":
                continue

        sources[label] = path
        print(f"  [OK] Added: {label} -> {path}")

    config["doc_sources"] = sources

    # --- Step 3: Verify paths ---
    header("Path Configuration")
    for key in ["chroma_dir", "symbol_index_path"]:
        val = config.get(key, "")
        print(f"  {key}: {val}")

    if config.get("chroma_dir", "").startswith("./"):
        abs_path = (Path(__file__).resolve().parent / config["chroma_dir"]).resolve()
        print(f"    (resolves to: {abs_path})")

    # --- Step 4: Write config ---
    with open(TARGET, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # --- Step 5: Ollama check ---
    header("Ollama Check")
    ollama_ok = check_ollama()

    # --- Done ---
    header("Setup Complete")
    print()
    print("  Next steps:")
    print()
    if not sources:
        print("  1. Add document sources to config.yaml:")
        print("     Edit the 'doc_sources' section in config.yaml")
        print()
        print("  2. Index your documents:")
        print("     python -m rag reindex")
    else:
        print("  1. Index your documents:")
        print("     python -m rag reindex")
    print()
    if not ollama_ok:
        print("  2. Start Ollama and pull the embedding model:")
        print("     ollama serve")
        print("     ollama pull nomic-embed-text")
    print("  3. Search your docs:")
    print("     python -m rag query \"your query here\"")
    print()
    print("  4. Run tests to verify:")
    print("     pytest tests/ -v -k \"not slow\"")
    print()
    print(f"  Config file: {TARGET}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
