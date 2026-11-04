"""PMC Open Access full-text fetcher.

For papers with a PMCID, fetches structured JATS XML from EFetch (db=pmc)
and extracts plain-text body sections, preserving section headers so the
chunker can tag section metadata. Papers without PMCID, or whose PMC fetch
fails, keep their abstract-only state.
"""

from __future__ import annotations

import asyncio

import httpx
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential

from medrag.config import get_settings
from medrag.models import Paper, PaperSource

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# PMC efetch tolerates batches but URL length is the binding constraint;
# 20 numeric IDs comfortably fits and keeps per-batch failures small.
_BATCH_SIZE = 20

# Block-level elements we do NOT want in retrieval text: tables, figures,
# captions, formulas, and bibliographic references are either noisy or
# better handled by other channels.
_SKIP_TAGS: frozenset[str] = frozenset(
    {
        "table-wrap",
        "table",
        "fig",
        "graphic",
        "media",
        "disp-formula",
        "inline-formula",
        "ref-list",
        "back",
    }
)


def _common_params() -> dict[str, str]:
    settings = get_settings()
    params = {"tool": "medrag", "email": settings.ncbi_email}
    if settings.ncbi_api_key is not None:
        params["api_key"] = settings.ncbi_api_key.get_secret_value()
    return params


def _normalize_pmcid(pmcid: str) -> str:
    """Strip the 'PMC' prefix so matching is consistent across PubMed/PMC."""
    stripped = pmcid.strip()
    return stripped[3:] if stripped.upper().startswith("PMC") else stripped


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _fetch_pmc_xml(client: httpx.AsyncClient, pmcids: list[str]) -> bytes:
    params = {
        **_common_params(),
        "db": "pmc",
        "id": ",".join(pmcids),
        "retmode": "xml",
    }
    resp = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
    resp.raise_for_status()
    return resp.content


async def enrich_with_full_text(papers: list[Paper], client: httpx.AsyncClient) -> list[Paper]:
    """Fetch PMC JATS XML for papers with a PMCID; return updated Paper list.

    Papers without a PMCID pass through unchanged. PMC fetch failures are
    swallowed per batch so one bad ID cannot drop the rest of the corpus —
    the affected papers simply stay abstract-only.
    """
    by_pmcid: dict[str, Paper] = {}
    for p in papers:
        if p.pmcid:
            by_pmcid[_normalize_pmcid(p.pmcid)] = p
    if not by_pmcid:
        return list(papers)

    full_text_by_pmcid: dict[str, str] = {}
    pmcids = list(by_pmcid.keys())
    for i in range(0, len(pmcids), _BATCH_SIZE):
        batch = pmcids[i : i + _BATCH_SIZE]
        try:
            xml_bytes = await _fetch_pmc_xml(client, batch)
        except (TimeoutError, httpx.HTTPError):
            continue
        full_text_by_pmcid.update(_parse_pmc_articleset(xml_bytes))
        await asyncio.sleep(0.15)  # gentle pacing under 10 req/s

    updated: list[Paper] = []
    for p in papers:
        if not p.pmcid:
            updated.append(p)
            continue
        text = full_text_by_pmcid.get(_normalize_pmcid(p.pmcid))
        if text:
            updated.append(p.model_copy(update={"full_text": text, "source": PaperSource.PMC_XML}))
        else:
            updated.append(p)
    return updated


# --------------------------------------------------------------------------- #
# JATS XML parsing                                                             #
# --------------------------------------------------------------------------- #


def _parse_pmc_articleset(xml_bytes: bytes) -> dict[str, str]:
    """Parse a <pmc-articleset> response into {normalized_pmcid: plain_text}."""
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return {}

    out: dict[str, str] = {}
    for article in _iter_local(root, "article"):
        pmcid = _article_pmcid(article)
        if not pmcid:
            continue
        body_text = _extract_body_text(article)
        if body_text:
            out[pmcid] = body_text
    return out


def _iter_local(element: etree._Element, localname: str) -> list[etree._Element]:
    """Find descendants by local tag name, ignoring any XML namespace."""
    return [el for el in element.iter() if etree.QName(el).localname == localname]


def _article_pmcid(article: etree._Element) -> str | None:
    for meta in _iter_local(article, "article-meta"):
        for aid in meta.iter():
            if etree.QName(aid).localname != "article-id":
                continue
            if aid.get("pub-id-type") == "pmc" and aid.text:
                return _normalize_pmcid(aid.text)
    return None


def _extract_body_text(article: etree._Element) -> str:
    bodies = _iter_local(article, "body")
    if not bodies:
        return ""
    lines: list[str] = []
    _render_body(bodies[0], lines)
    return "\n".join(line for line in lines if line.strip())


def _render_body(element: etree._Element, lines: list[str]) -> None:
    """Walk a JATS <body> emitting section titles and paragraph text."""
    for child in element:
        tag = etree.QName(child).localname
        if tag in _SKIP_TAGS:
            continue
        if tag == "sec":
            title_el = next(
                (c for c in child if etree.QName(c).localname == "title"),
                None,
            )
            if title_el is not None:
                title = _inline_text(title_el).strip()
                if title:
                    lines.append("")
                    lines.append(title)
            _render_body(child, lines)
        elif tag == "p":
            text = _inline_text(child).strip()
            if text:
                lines.append(text)
        elif tag == "title":
            # Titles are rendered when their parent <sec> is processed.
            continue
        else:
            # Unknown container: recurse so we don't drop nested paragraphs.
            _render_body(child, lines)


def _inline_text(element: etree._Element) -> str:
    """Gather visible text from a JATS element, skipping xrefs/tables/figures."""
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        tag = etree.QName(child).localname
        if tag in _SKIP_TAGS or tag == "xref":
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_inline_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)
