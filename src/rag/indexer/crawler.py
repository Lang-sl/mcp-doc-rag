from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterator

from rag.indexer.parser_registry import classify_file  # noqa: E402 — registration side-effect


@dataclass
class FileEntry:
    abs_path: str
    rel_path: str
    source_label: str
    source_module: str
    file_type: str  # "html", "pdf", "header", "unknown"
    needs_index: bool
    sha1: str | None = None  # cached to avoid re-reading in update_state


def _detect_module(rel_path: str) -> str:
    """Extract sub-module from the first directory component of *rel_path*.

    Backslashes are normalised to forward slashes.  If the path has no
    directory component the module is ``"root"``.
    """
    normalised = rel_path.replace("\\", "/")
    parts = normalised.split("/")
    if len(parts) <= 1:
        return "root"
    return parts[0]


def _load_state(state_path: str) -> dict:
    """Load index state from a JSON file.

    Returns ``{source_label: {rel_path: {sha1, mtime, size}}}``.
    If the file does not exist an empty dict is returned.
    """
    if not os.path.isfile(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_state(state_path: str, state: dict) -> None:
    """Persist *state* to a JSON file, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)


def _file_changed(abs_path: str, cached: dict | None) -> tuple[bool, str | None]:
    """Three-tier check whether *abs_path* has changed since it was cached.

    Returns ``(needs_index, sha1_hex | None)``.

    *needs_index* is ``True`` when:
    - *cached* is ``None`` (never seen before).
    - ``os.stat`` fails — the file may be unreadable or gone.
    - The cached ``sha1`` does not match the current content hash.

    Returns ``(False, cached_sha1)`` (fast path) when ``mtime`` **and**
    ``size`` are both unchanged — the file is not read at all.
    """
    if cached is None:
        return True, None

    try:
        st = os.stat(abs_path)
    except OSError:
        return True, None

    if st.st_mtime_ns == cached["mtime"] and st.st_size == cached["size"]:
        return False, cached.get("sha1")

    sha1 = hashlib.sha1()
    try:
        with open(abs_path, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                sha1.update(chunk)
    except OSError:
        return True, None

    new_digest = sha1.hexdigest()
    return new_digest != cached["sha1"], new_digest


def crawl_source(
    source_label: str, source_path: str, state_path: str
) -> Iterator[FileEntry]:
    """Walk a single source directory and yield :class:`FileEntry` items.

    Files classified as ``"unknown"`` are skipped.  *state_path* is read via
    :func:`_load_state` to decide whether each file needs indexing.
    """
    state = _load_state(state_path)
    source_state = state.get(source_label, {})

    for root, _dirs, files in os.walk(source_path):
        for filename in files:
            abs_path = os.path.join(root, filename)
            file_type = classify_file(filename)
            if file_type == "unknown":
                continue

            rel_path = os.path.relpath(abs_path, source_path)
            module = _detect_module(rel_path)
            cached = source_state.get(rel_path)
            needs_index, sha1 = _file_changed(abs_path, cached)

            yield FileEntry(
                abs_path=abs_path,
                rel_path=rel_path,
                source_label=source_label,
                source_module=module,
                file_type=file_type,
                needs_index=needs_index,
                sha1=sha1,
            )


def update_state(
    state_path: str, source_label: str, entries: list[FileEntry]
) -> None:
    """Update the index state on disk for every entry in *entries*.

    For each file the current ``sha1``, ``mtime`` (nanosecond precision) and
    ``size`` are recorded under *source_label* in the state JSON.

    Uses ``entry.sha1`` when available (from :func:`crawl_source`) to avoid
    re-reading the file.  Falls back to a fresh ``sha1`` computation only
    when the cached digest is missing.
    """
    state = _load_state(state_path)
    source_state: dict[str, dict] = state.setdefault(source_label, {})

    for entry in entries:
        digest = entry.sha1
        if digest is None:
            sha1_obj = hashlib.sha1()
            try:
                with open(entry.abs_path, "rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        sha1_obj.update(chunk)
                digest = sha1_obj.hexdigest()
            except OSError:
                continue

        try:
            st = os.stat(entry.abs_path)
        except OSError:
            continue

        source_state[entry.rel_path] = {
            "sha1": digest,
            "mtime": st.st_mtime_ns,
            "size": st.st_size,
        }

    _save_state(state_path, state)


def detect_deleted_files(
    source_label: str, source_path: str, state_path: str
) -> list[str]:
    """Return absolute paths of files that are in the index state but no
    longer exist on disk for *source_label*.

    Callers should clean up the returned files from ChromaDB, the symbol
    index, and the state file.
    """
    state = _load_state(state_path)
    source_state = state.get(source_label, {})
    deleted: list[str] = []
    for rel_path in source_state:
        abs_path = os.path.join(source_path, rel_path)
        if not os.path.isfile(abs_path):
            deleted.append(abs_path)
    return deleted


def remove_deleted_from_state(
    state_path: str, source_label: str, source_path: str, deleted_abs_paths: list[str]
) -> int:
    """Remove deleted-file entries from the index state.

    Returns the number of entries removed.
    """
    state = _load_state(state_path)
    source_state = state.get(source_label, {})
    removed = 0
    for abs_path in deleted_abs_paths:
        # Convert abs_path back to rel_path for state lookup
        try:
            rel_path = os.path.relpath(abs_path, source_path)
        except ValueError:
            continue
        if rel_path in source_state:
            del source_state[rel_path]
            removed += 1
    if removed:
        _save_state(state_path, state)
    return removed


def crawl_all(
    doc_sources: dict[str, str], state_path: str
) -> Iterator[FileEntry]:
    """Iterate all document sources and yield :class:`FileEntry` items.

    *doc_sources* maps ``source_label`` to ``source_path``.
    """
    for source_label, source_path in doc_sources.items():
        yield from crawl_source(source_label, source_path, state_path)
