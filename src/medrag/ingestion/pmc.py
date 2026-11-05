"""PMC Open Access full-text fetcher.

For papers with a PMCID, fetches structured JATS XML from EFetch (db=pmc)
and extracts plain-text body sections, preserving section headers so the
chunker can tag section metadata. Papers without PMCID, or whose PMC fetch
fails, keep their abstract-only state.
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

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

# `pub-id-type` values that carry an article's PMC ID, most specific first.
# PMC now tags it "pmcid" with a "PMC"-prefixed value; it used to be "pmc" with
# the bare number, and "pmcaid" is the bare number under the current scheme.
# Matching only the old value made every article look unidentified, so a run
# fetched every body and kept none of them.
_PMCID_ID_TYPES: tuple[str, ...] = ("pmcid", "pmc", "pmcaid")


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


# reraise=True: after the last attempt, raise the httpx error itself and not
# tenacity's RetryError, so `enrich_with_full_text` can skip the batch.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
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
    the affected papers simply stay abstract-only. So does a paper whose
    PMCID turns out to be another paper's: the PMC article states its own
    PMID, and a body is only attached when that matches.
    """
    by_pmcid: dict[str, Paper] = {}
    for p in papers:
        if p.pmcid:
            by_pmcid[_normalize_pmcid(p.pmcid)] = p
    if not by_pmcid:
        return list(papers)

    articles: dict[str, _PmcArticle] = {}
    pmcids = list(by_pmcid.keys())
    for i in range(0, len(pmcids), _BATCH_SIZE):
        batch = pmcids[i : i + _BATCH_SIZE]
        try:
            xml_bytes = await _fetch_pmc_xml(client, batch)
        except (TimeoutError, httpx.HTTPError):
            continue
        articles.update(_parse_pmc_articleset(xml_bytes))
        await asyncio.sleep(0.15)  # gentle pacing under 10 req/s

    updated: list[Paper] = []
    for p in papers:
        if not p.pmcid:
            updated.append(p)
            continue
        article = articles.get(_normalize_pmcid(p.pmcid))
        if article is not None and article.pmid in (None, p.pmid):
            updated.append(
                p.model_copy(update={"full_text": article.text, "source": PaperSource.PMC_XML})
            )
        else:
            updated.append(p)
    return updated


# --------------------------------------------------------------------------- #
# JATS XML parsing                                                             #
# --------------------------------------------------------------------------- #


class _PmcArticle(NamedTuple):
    pmid: str | None  # the PMID the PMC record states for itself, if any
    text: str  # plain-text body


def _parse_pmc_articleset(xml_bytes: bytes) -> dict[str, _PmcArticle]:
    """Parse a <pmc-articleset> response into {normalized_pmcid: article}.

    Articles with no body text are left out. PMC returns front matter only
    when the publisher does not allow full-text XML download.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return {}

    out: dict[str, _PmcArticle] = {}
    for article in _iter_local(root, "article"):
        ids = _article_ids(article)
        pmcid = next((ids[t] for t in _PMCID_ID_TYPES if t in ids), None)
        if not pmcid:
            continue
        body_text = _extract_body_text(article)
        if body_text:
            out[_normalize_pmcid(pmcid)] = _PmcArticle(pmid=ids.get("pmid"), text=body_text)
    return out


def _local(element: etree._Element) -> str | None:
    """Local tag name without its namespace; None for comments and PIs.

    PMC XML carries processing instructions and comments, whose `tag` is not
    a string, and `etree.QName` raises on those.
    """
    return etree.QName(element).localname if isinstance(element.tag, str) else None


def _iter_local(element: etree._Element, localname: str) -> list[etree._Element]:
    """Find descendants by local tag name, ignoring any XML namespace."""
    return [el for el in element.iter() if _local(el) == localname]


def _article_ids(article: etree._Element) -> dict[str, str]:
    """The article's own identifiers, keyed by `pub-id-type`."""
    metas = _iter_local(article, "article-meta")
    if not metas:
        return {}
    # The first <article-meta> is the article's own; later ones belong to
    # sub-articles such as a translation or a reviewer report.
    by_type: dict[str, str] = {}
    for aid in _iter_local(metas[0], "article-id"):
        id_type = aid.get("pub-id-type")
        if id_type and aid.text and aid.text.strip():
            by_type.setdefault(id_type, aid.text.strip())
    return by_type


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
        tag = _local(child)
        if tag is None or tag in _SKIP_TAGS:
            continue
        if tag == "sec":
            title_el = next((c for c in child if _local(c) == "title"), None)
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
        tag = _local(child)
        if tag is None or tag in _SKIP_TAGS or tag == "xref":
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_inline_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)
