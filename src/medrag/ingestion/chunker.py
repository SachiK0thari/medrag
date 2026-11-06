"""Text chunking.

Recursive character splitter with a rough token estimate (chars / 4).
Keeps section metadata when PMC full text is available; otherwise chunks the
abstract as a single section.

A chunk is never longer than the target plus the overlap, whatever the text
looks like: a paragraph longer than the target is split again on the next
finer separator.

That bound is in characters, and the embedder's limit is in tokens. Prose runs
at about 4.5 characters per token, but a results paragraph full of numbers and
confidence intervals runs nearer 2.5, so a chunk inside the character bound can
still be over the embedder's 512 tokens, and the embedder truncates without
saying so. Pass `count_tokens` and `max_tokens` to `chunk_paper` (ingestion
passes the embedding model's own tokenizer) and any such chunk is halved until
it fits.

We deliberately avoid a heavy dependency like LangChain's text splitters —
the splitter is about a hundred lines and trivially testable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable

from medrag.config import get_settings
from medrag.models import Chunk, Paper

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Packed text shorter than this is not closed off as a chunk of its own when
# the next part does not fit. In article bodies it is nearly always a
# sub-heading ("Study population"), which is useless alone and useful as the
# first line of the paragraph it introduces.
_MIN_STANDALONE_CHARS = 100

# A piece shorter than this is dropped: what is left is a heading whose
# section held only a table or a figure, which the PMC parser skips.
_MIN_CHUNK_CHARS = 50

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English prose."""
    return max(1, len(text) // 4)


def _chunk_id(pmid: str, section: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{pmid}:{section}:{idx}:{text[:64]}".encode()).hexdigest()[:16]
    return f"{pmid}-{idx:03d}-{h}"


def _recursive_split(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split `text` into pieces of ~target_chars, trying separators in order.

    Every piece is at most `target_chars` long before the overlap is added.
    """
    chunks = _split(text.strip(), target_chars, _SEPARATORS)
    if overlap_chars > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap_chars)
    return chunks


def _split(text: str, target_chars: int, separators: list[str]) -> list[str]:
    """Split on the coarsest separator present, packing parts up to the target.

    A part that is still too long on its own (one long paragraph, one long
    sentence) is split again with the finer separators that remain.
    """
    if len(text) <= target_chars:
        return [text] if text.strip() else []

    for i, sep in enumerate(separators):
        if sep == "":
            # Nothing left to split on: hard wrap.
            return [text[j : j + target_chars] for j in range(0, len(text), target_chars)]
        if sep not in text:
            continue
        chunks: list[str] = []
        buf = ""
        for part in text.split(sep):
            candidate = buf + (sep if buf else "") + part
            if len(candidate) <= target_chars:
                buf = candidate
                continue
            if buf and len(buf) >= _MIN_STANDALONE_CHARS:
                chunks.append(buf)
                buf = ""
            # A short `buf` is a sub-heading; it stays with the text it introduces.
            rest = buf + (sep if buf else "") + part
            if len(rest) <= target_chars:
                buf = rest
            else:
                # Keep the last sub-piece open so the next part can join it.
                *done, buf = _split(rest, target_chars, separators[i + 1 :])
                chunks.extend(done)
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return [text]


def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap_chars:]
        out.append(tail + " " + chunks[i])
    return out


def _fit_token_limit(piece: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Halve `piece` at separator boundaries until every part is within `max_tokens`."""
    if len(piece) <= 1 or count_tokens(piece) <= max_tokens:
        return [piece]
    # Aim past the midpoint: packing stops at the last separator before the
    # target, and an exact half would leave a third piece of a few words.
    halves = _split(piece, len(piece) * 3 // 5 + 1, _SEPARATORS)
    return [part for half in halves for part in _fit_token_limit(half, count_tokens, max_tokens)]


def chunk_paper(
    paper: Paper,
    *,
    count_tokens: TokenCounter | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Produce retrievable chunks from a Paper.

    With `count_tokens` and `max_tokens`, no chunk exceeds `max_tokens` as
    counted by `count_tokens`, and `n_tokens` is that count instead of the
    character estimate.
    """
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
    for section_name, body in sections:
        pieces = _recursive_split(body, target_chars, overlap_chars)
        if count_tokens is not None and max_tokens is not None:
            pieces = [
                part
                for piece in pieces
                for part in _fit_token_limit(piece, count_tokens, max_tokens)
            ]
        for piece in pieces:
            if len(piece) < _MIN_CHUNK_CHARS:
                continue
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(paper.pmid, section_name, idx, piece),
                    pmid=paper.pmid,
                    text=piece,
                    section=section_name,
                    chunk_index=idx,
                    n_tokens=count_tokens(piece) if count_tokens else estimate_tokens(piece),
                )
            )
    return chunks


# A top-level section title on a line of its own, with or without a number
# ("Methods", "2. Methods", "MATERIALS AND METHODS").
_SECTION_HEADER_RE = re.compile(
    r"^[ \t]*(?:\d{1,2}[.)]?[ \t]+)?"
    r"(abstract|introduction(?: and background)?|background|methods?|methodology|"
    r"materials? and methods|research design and methods|patients and methods|"
    r"results?|discussions?|conclusions?|references)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Substring of a matched title -> the section name stored on the chunk.
_SECTION_NAMES: tuple[tuple[str, str], ...] = (
    ("method", "methods"),
    ("introduction", "introduction"),
    ("background", "background"),
    ("result", "results"),
    ("discussion", "discussion"),
    ("conclusion", "conclusion"),
    ("abstract", "abstract"),
    ("references", "references"),
)


def _section_name(title: str) -> str:
    lowered = title.lower()
    return next(name for key, name in _SECTION_NAMES if key in lowered)


def _split_by_section(full_text: str) -> Iterable[tuple[str, str]]:
    """Split plain-text full text on IMRaD-style headers.

    Text with no recognised header is one 'body' section, and so is any text
    that comes before the first header (many articles open with an untitled
    introduction).
    """
    matches = list(_SECTION_HEADER_RE.finditer(full_text))
    lead = full_text[: matches[0].start()].strip() if matches else full_text.strip()
    if lead:
        yield ("body", lead)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[m.end() : end].strip()
        if body:
            yield (_section_name(m.group(1)), body)
