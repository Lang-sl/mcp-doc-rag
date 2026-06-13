#!/usr/bin/env python3
"""Interactive configuration setup for mcp-doc-rag.

Run this script once after cloning the repository:

    python setup_config.py

It will:
  1. Create config.yaml from the template
  2. Help you configure document sources
  3. Optionally create gateway.yaml for CodeGraph gateway search
  4. Verify Ollama is available (optional)
  5. Print next steps
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).resolve().parent / "src" / "rag" / "config.example.yaml"
TARGET = Path(__file__).resolve().parent / "config.yaml"
GATEWAY_TEMPLATE = Path(__file__).resolve().parent / "src" / "rag" / "gateway.example.yaml"
GATEWAY_TARGET = Path(__file__).resolve().parent / "gateway.yaml"


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


def configure_gateway() -> bool:
    """Optionally create gateway.yaml for daemon-backed MCP gateway mode."""
    header("MCP Gateway")
    print("\n  The gateway provides daemon-backed MCP access to doc-rag tools.")
    print("  It can optionally include CodeGraph for combined doc+code search.")
    print("  Skip this if you only want the standalone doc-rag MCP server.\n")

    create = ask("Create gateway.yaml for daemon-backed MCP gateway mode? (y/n)", "y")
    if create.lower() != "y":
        print("  Skipping gateway.yaml")
        return False

    if GATEWAY_TARGET.exists():
        print(f"\n  gateway.yaml already exists at: {GATEWAY_TARGET}")
        overwrite = ask("Overwrite? (y/n)", "n")
        if overwrite.lower() != "y":
            print("  Keeping existing gateway.yaml")
            return True

    shutil.copy(GATEWAY_TEMPLATE, GATEWAY_TARGET)

    with open(GATEWAY_TARGET, "r", encoding="utf-8") as fh:
        gateway_config = yaml.safe_load(fh)
    if not isinstance(gateway_config, dict):
        gateway_config = {}

    gateway_config["doc_rag"] = {"config_path": str(TARGET.resolve())}
    gateway_config["daemon"] = {
        "autostart": True,
        "host": "127.0.0.1",
        "port": 0,
    }

    enable_codegraph = ask("Enable optional CodeGraph integration? (y/n)", "n")
    if enable_codegraph.lower() != "y":
        gateway_config.pop("codegraph", None)
    else:
        codegraph = gateway_config.get("codegraph")
        if not isinstance(codegraph, dict):
            codegraph = {}

        code_project = ask("Code project path for CodeGraph (leave blank to fill later)")
        if code_project:
            if not os.path.isdir(code_project):
                print(f"  [WARN] Directory not found: {code_project}")
                still_add = ask("  Add anyway? (y/n)", "n")
                if still_add.lower() == "y":
                    codegraph["cwd"] = code_project
            else:
                codegraph["cwd"] = code_project

        codegraph.setdefault("command", "npx")
        codegraph.setdefault("args", ["-y", "@colbymchenry/codegraph@0.9.9", "serve", "--mcp"])
        codegraph.setdefault("cwd", "<absolute-path-to-code-project>")
        gateway_config["codegraph"] = codegraph

    with open(GATEWAY_TARGET, "w", encoding="utf-8") as fh:
        yaml.safe_dump(gateway_config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  [OK] Created: {GATEWAY_TARGET}")
    return True


def main() -> int:
    header("mcp-doc-rag Configuration Setup")

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

    # --- Step 5: Optional gateway config ---
    gateway_configured = configure_gateway()

    # --- Step 6: Ollama check ---
    header("Ollama Check")
    ollama_ok = check_ollama()

    # --- Done ---
    header("Setup Complete")
    print()
    print("  Next steps:")
    print()
    step = 1
    if not sources:
        print(f"  {step}. Add document sources to config.yaml:")
        print("     Edit the 'doc_sources' section in config.yaml")
        print()
        step += 1
    print(f"  {step}. Index your documents:")
    print("     python -m rag reindex")
    step += 1
    print()
    if not ollama_ok:
        print(f"  {step}. Start Ollama and pull the embedding model:")
        print("     ollama serve")
        print("     ollama pull nomic-embed-text")
        step += 1
    print(f"  {step}. Search your docs:")
    print("     python -m rag query \"your query here\"")
    step += 1
    print()
    if gateway_configured:
        print(f"  {step}. Start the gateway adapter:")
        print("     python -m rag adapter")
        step += 1
        print()
        print(f"  {step}. Or use the stdio fallback:")
        print("     python -m rag gateway")
        step += 1
        print()
    print(f"  {step}. Run tests to verify:")
    print("     pytest tests/ -v -k \"not slow\"")
    print()
    print(f"  Config file: {TARGET}")
    if gateway_configured:
        print(f"  Gateway config file: {GATEWAY_TARGET}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
