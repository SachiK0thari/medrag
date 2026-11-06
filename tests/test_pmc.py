"""Tests for the PMC full-text fetcher. No network: responses are canned."""

from __future__ import annotations

import httpx
import pytest
from tenacity import wait_none

from medrag.ingestion.pmc import _fetch_pmc_xml, _parse_pmc_articleset, enrich_with_full_text
from medrag.models import Paper, PaperSource

_BODY = """
    <body>
      <?properties open_access?>
      <sec>
        <title>Introduction</title>
        <!-- typesetter note -->
        <p>GLP-1 receptor agonists reduce cardiovascular events<xref rid="r1">1</xref> in
        people with type 2 diabetes.<?pagebreak?> The effect is consistent.</p>
      </sec>
      <sec>
        <title>Results</title>
        <p>MACE occurred in 13.0% versus 14.9%.</p>
        <table-wrap><table><tr><td>not retrieval text</td></tr></table></table-wrap>
      </sec>
    </body>
"""


def _article(article_ids: str, body: str = _BODY) -> str:
    return f"""
  <article xmlns:xlink="http://www.w3.org/1999/xlink">
    <front><article-meta>{article_ids}</article-meta></front>
    {body}
  </article>"""


def _articleset(*articles: str) -> bytes:
    return f"<pmc-articleset>{''.join(articles)}</pmc-articleset>".encode()


# The shape efetch (db=pmc) returns today: the PMC ID is tagged "pmcid" and
# carries the "PMC" prefix. Captured from a live response and trimmed.
_CURRENT_IDS = """
      <article-id pub-id-type="pmcid">PMC1234567</article-id>
      <article-id pub-id-type="pmcid-ver">PMC1234567.1</article-id>
      <article-id pub-id-type="pmcaid">1234567</article-id>
      <article-id pub-id-type="pmid">12345678</article-id>
      <article-id pub-id-type="doi">10.1056/NEJMoa1603827</article-id>
"""
# The shape the parser was first written against.
_LEGACY_IDS = """
      <article-id pub-id-type="pmid">12345678</article-id>
      <article-id pub-id-type="pmc">1234567</article-id>
"""


def test_parses_current_pmcid_tagging() -> None:
    parsed = _parse_pmc_articleset(_articleset(_article(_CURRENT_IDS)))
    assert list(parsed) == ["1234567"]


def test_parses_legacy_pmc_tagging() -> None:
    parsed = _parse_pmc_articleset(_articleset(_article(_LEGACY_IDS)))
    assert list(parsed) == ["1234567"]


def test_falls_back_to_pmcaid() -> None:
    ids = '<article-id pub-id-type="pmcaid">1234567</article-id>'
    assert list(_parse_pmc_articleset(_articleset(_article(ids)))) == ["1234567"]


def test_body_text_keeps_section_titles_and_skips_noise() -> None:
    article = _parse_pmc_articleset(_articleset(_article(_CURRENT_IDS)))["1234567"]
    assert article.pmid == "12345678"
    text = article.text
    lines = text.split("\n")
    assert lines[0] == "Introduction"
    assert "Results" in lines
    assert "MACE occurred in 13.0% versus 14.9%." in lines
    # Citation markers, tables, comments and processing instructions are dropped,
    # and the text on either side of them is kept.
    assert "events in" in text
    assert "The effect is consistent." in text
    assert "not retrieval text" not in text
    assert "typesetter" not in text


def test_a_paragraph_is_one_line_however_the_xml_is_wrapped() -> None:
    """The chunker reads a line break as a paragraph boundary.

    The fixture's first paragraph breaks after "in", as pretty-printed PMC XML
    does, and that break used to survive into the text.
    """
    article = _parse_pmc_articleset(_articleset(_article(_CURRENT_IDS)))["1234567"]
    assert article.text.split("\n") == [
        "Introduction",
        "GLP-1 receptor agonists reduce cardiovascular events in people with type 2 "
        "diabetes. The effect is consistent.",
        "Results",
        "MACE occurred in 13.0% versus 14.9%.",
    ]


def test_brackets_emptied_by_dropped_citations_are_removed() -> None:
    body = """
    <body><sec><title>Discussion</title>
      <p>Seen in earlier trials [<xref rid="r1">1</xref>, <xref rid="r2">2</xref>&#8211;<xref
      rid="r4">4</xref>] and in a registry (<xref rid="r5">5</xref>). MACE fell
      (HR 0.74 [95% CI 0.58-0.95]).</p>
    </sec></body>"""
    article = _parse_pmc_articleset(_articleset(_article(_CURRENT_IDS, body=body)))["1234567"]
    assert article.text.split("\n")[1] == (
        "Seen in earlier trials and in a registry. MACE fell (HR 0.74 [95% CI 0.58-0.95])."
    )


def test_article_without_body_or_id_is_skipped() -> None:
    no_body = _article(_CURRENT_IDS, body="")
    no_id = _article('<article-id pub-id-type="doi">10.1/x</article-id>')
    assert _parse_pmc_articleset(_articleset(no_body, no_id)) == {}


def test_sub_article_id_does_not_replace_the_articles_own() -> None:
    sub = """
    <sub-article><front-stub><article-meta>
      <article-id pub-id-type="pmcid">PMC9999999</article-id>
    </article-meta></front-stub></sub-article>"""
    parsed = _parse_pmc_articleset(_articleset(_article(_CURRENT_IDS, body=_BODY + sub)))
    assert list(parsed) == ["1234567"]


def test_malformed_xml_returns_nothing() -> None:
    assert _parse_pmc_articleset(b"<pmc-articleset><article>") == {}


async def test_enrich_attaches_full_text_to_the_matching_paper(sample_paper: Paper) -> None:
    other = sample_paper.model_copy(update={"pmid": "2", "pmcid": "PMC7654321"})
    no_pmcid = sample_paper.model_copy(update={"pmid": "3", "pmcid": None})
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.params["id"])
        return httpx.Response(200, content=_articleset(_article(_CURRENT_IDS)))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        enriched = await enrich_with_full_text([sample_paper, other, no_pmcid], client)

    assert requested == ["1234567,7654321"]
    assert [p.pmid for p in enriched] == [sample_paper.pmid, "2", "3"]
    assert enriched[0].source is PaperSource.PMC_XML
    assert enriched[0].full_text is not None
    assert enriched[0].full_text.startswith("Introduction")
    # PMC returned no article for the second ID, and the third has none to ask for.
    assert enriched[1].full_text is None
    assert enriched[2].full_text is None


async def test_enrich_refuses_a_body_that_belongs_to_another_paper(sample_paper: Paper) -> None:
    """A stored PMC ID can be a cited paper's; PMC's own PMID gives that away."""
    citing = sample_paper.model_copy(update={"pmid": "99999999"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_articleset(_article(_CURRENT_IDS)))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        enriched = await enrich_with_full_text([citing], client)

    assert enriched == [citing]


async def test_enrich_survives_a_failed_batch(
    sample_paper: Paper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch that fails every retry is skipped; its papers stay abstract-only."""
    monkeypatch.setattr(_fetch_pmc_xml.retry, "wait", wait_none())  # type: ignore[attr-defined]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        enriched = await enrich_with_full_text([sample_paper], client)

    assert calls == 3
    assert enriched == [sample_paper]
