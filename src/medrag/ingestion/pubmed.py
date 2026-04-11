"""PubMed E-utilities client.

Uses esearch -> efetch flow to get PMIDs and Medline records for a topic query.
Respects NCBI's 3 req/s (10 req/s with API key) rate limit via tenacity backoff.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential

from medrag.config import get_settings
from medrag.models import Paper, PaperSource

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    """Async client for PubMed E-utilities."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._settings = get_settings()
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None

    async def __aenter__(self) -> PubMedClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _common_params(self) -> dict[str, str]:
        params = {"tool": "medrag", "email": self._settings.ncbi_email}
        if self._settings.ncbi_api_key is not None:
            params["api_key"] = self._settings.ncbi_api_key.get_secret_value()
        return params

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search(self, query: str, max_results: int = 500) -> list[str]:
        """esearch -> list of PMIDs matching `query`, newest first."""
        params: dict[str, Any] = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "pub_date",
            "retmode": "json",
        }
        resp = await self._client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        return list(data["esearchresult"]["idlist"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def fetch_metadata(self, pmids: list[str]) -> list[Paper]:
        """efetch (db=pubmed, retmode=xml) -> Paper records (no PMC full text yet)."""
        if not pmids:
            return []
        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        resp = await self._client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        resp.raise_for_status()
        return _parse_pubmed_xml(resp.content)

    async def fetch_in_batches(self, pmids: list[str], batch_size: int = 100) -> list[Paper]:
        """Batch metadata fetches to avoid URL length limits."""
        papers: list[Paper] = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            papers.extend(await self.fetch_metadata(batch))
            await asyncio.sleep(0.15)  # gentle pacing under 10 req/s
        return papers


def _parse_pubmed_xml(xml_bytes: bytes) -> list[Paper]:
    """Parse a PubmedArticleSet XML blob into Paper records."""
    root = etree.fromstring(xml_bytes)
    papers: list[Paper] = []
    for art in root.iter("PubmedArticle"):
        try:
            papers.append(_paper_from_article(art))
        except Exception:
            continue
    return papers


def _paper_from_article(art: etree._Element) -> Paper:
    pmid_el = art.find(".//PMID")
    pmid = pmid_el.text if pmid_el is not None else None
    if not pmid:
        raise ValueError("missing PMID")

    title_el = art.find(".//ArticleTitle")
    title = "".join(title_el.itertext()) if title_el is not None else "Untitled"

    abstract_parts = ["".join(el.itertext()) for el in art.findall(".//Abstract/AbstractText")]
    abstract = "\n".join(p for p in abstract_parts if p) or None

    authors: list[str] = []
    for author in art.findall(".//AuthorList/Author"):
        last = author.findtext("LastName")
        fore = author.findtext("ForeName")
        if last:
            authors.append(f"{fore} {last}".strip() if fore else last)

    journal = art.findtext(".//Journal/Title")
    year_text = art.findtext(".//JournalIssue/PubDate/Year")
    year = int(year_text) if year_text and year_text.isdigit() else None

    doi = None
    for aid in art.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text
            break

    pmcid = None
    for aid in art.findall(".//ArticleId"):
        if aid.get("IdType") == "pmc":
            pmcid = aid.text
            break

    mesh_terms = [el.text or "" for el in art.findall(".//MeshHeading/DescriptorName") if el.text]

    return Paper(
        pmid=pmid,
        pmcid=pmcid,
        title=title,
        abstract=abstract,
        full_text=None,  # filled in later by PMC fetcher if available
        authors=authors,
        journal=journal,
        year=year,
        doi=doi,
        mesh_terms=mesh_terms,
        source=PaperSource.PUBMED_ABSTRACT,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",  # type: ignore[arg-type]
    )
