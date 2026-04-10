"""PMC Open Access full-text fetcher.

For papers with a PMCID, fetches structured JATS XML from EFetch (db=pmc)
and extracts plain-text body sections. Papers without PMCID keep abstract-only.
"""

from __future__ import annotations

import httpx

from medrag.models import Paper

# TODO(saturday): implement using efetch.fcgi?db=pmc&id=<PMCID>&retmode=xml
# Parse <body><sec>...<p>...</p></sec></body> into plain text, preserving section
# headers so the chunker can tag section metadata.


async def enrich_with_full_text(
    papers: list[Paper], client: httpx.AsyncClient
) -> list[Paper]:
    """Fetch PMC JATS XML for papers with a PMCID; return updated Paper list."""
    raise NotImplementedError("implement Saturday morning")
