"""Text chunking.

Recursive character splitter with a rough token estimate (chars / 4).
Keeps section metadata when PMC full text is available; otherwise chunks the
abstract as a single section.

We deliberately avoid a heavy dependency like LangChain's text splitters —
this is ~50 lines and trivially testable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from medrag.config import get_settings
from medrag.models import Chunk, Paper

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English prose."""
    return max(1, len(text) // 4)


def _chunk_id(pmid: str, section: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{pmid}:{section}:{idx}:{text[:64]}".encode()).hexdigest()[:16]
    return f"{pmid}-{idx:03d}-{h}"


def _recursive_split(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split `text` into pieces of ~target_chars, trying separators in order."""
    text = text.strip()
    if len(text) <= target_chars:
        return [text] if text else []

    for sep in _SEPARATORS:
        if sep == "":
            # Hard wrap fallback
            return [
                text[i : i + target_chars]
                for i in range(0, len(text), target_chars - overlap_chars)
            ]
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks: list[str] = []
        buf = ""
        for part in parts:
            candidate = buf + (sep if buf else "") + part
            if len(candidate) <= target_chars:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
        # Add overlap
        if overlap_chars > 0 and len(chunks) > 1:
            chunks = _apply_overlap(chunks, overlap_chars)
        return [c for c in chunks if c.strip()]
    return [text]


def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap_chars:]
        out.append(tail + " " + chunks[i])
    return out


def chunk_paper(paper: Paper) -> list[Chunk]:
    """Produce retrievable chunks from a Paper."""
    settings = get_settings()
    target_chars = settings.chunk_size_tokens * 4
    overlap_chars = settings.chunk_overlap_tokens * 4

    sections: list[tuple[str, str]] = []
    if paper.abstract:
        sections.append(("abstract", paper.abstract))
    if paper.full_text:
        for name, body in _split_by_section(paper.full_text):
            sections.append((name, body))

    chunks: list[Chunk] = []
    global_idx = 0
    for section_name, body in sections:
        for piece in _recursive_split(body, target_chars, overlap_chars):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(paper.pmid, section_name, global_idx, piece),
                    pmid=paper.pmid,
                    text=piece,
                    section=section_name,
                    chunk_index=global_idx,
                    n_tokens=estimate_tokens(piece),
                )
            )
            global_idx += 1
    return chunks


_SECTION_HEADER_RE = re.compile(
    r"^\s*(abstract|introduction|background|methods?|materials and methods|"
    r"results|discussion|conclusions?|references)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_by_section(full_text: str) -> Iterable[tuple[str, str]]:
    """Split plain-text full text on IMRaD-style headers. Fallback = single 'body'."""
    matches = list(_SECTION_HEADER_RE.finditer(full_text))
    if not matches:
        yield ("body", full_text)
        return
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_name = m.group(1).lower().replace(" ", "_")
        body = full_text[start:end].strip()
        if body:
            yield (section_name, body)
