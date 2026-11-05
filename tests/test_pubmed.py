"""Tests for PubMed record parsing. No network: the XML is inline."""

from __future__ import annotations

from medrag.ingestion.pubmed import _parse_pubmed_xml
from medrag.models import Paper, PaperSource

_REFERENCES = """
        <ReferenceList>
          <Reference>
            <Citation>Marso SP, et al. N Engl J Med. 2016;375(4):311-322.</Citation>
            <ArticleIdList>
              <ArticleId IdType="doi">10.1056/NEJMoa1603827</ArticleId>
              <ArticleId IdType="pmc">PMC4985288</ArticleId>
              <ArticleId IdType="pubmed">27295427</ArticleId>
            </ArticleIdList>
          </Reference>
        </ReferenceList>"""


def _record(pmid: str, own_ids: str) -> str:
    return f"""
    <PubmedArticle>
      <MedlineCitation>
        <PMID Version="1">{pmid}</PMID>
        <Article>
          <Journal>
            <JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue>
            <Title>Diabetologia</Title>
          </Journal>
          <ArticleTitle>Semaglutide and <i>cardiovascular</i> outcomes</ArticleTitle>
          <Abstract>
            <AbstractText Label="AIMS">We assessed MACE.</AbstractText>
            <AbstractText Label="RESULTS">MACE fell.</AbstractText>
          </Abstract>
          <AuthorList>
            <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
            <Author><LastName>Consortium</LastName></Author>
          </AuthorList>
        </Article>
        <MeshHeadingList>
          <MeshHeading><DescriptorName>Glucagon-Like Peptide 1</DescriptorName></MeshHeading>
        </MeshHeadingList>
      </MedlineCitation>
      <PubmedData>
        <ArticleIdList>{own_ids}</ArticleIdList>{_REFERENCES}
      </PubmedData>
    </PubmedArticle>"""


def _parse(*records: str) -> list[Paper]:
    return _parse_pubmed_xml(f"<PubmedArticleSet>{''.join(records)}</PubmedArticleSet>".encode())


def test_parses_core_fields() -> None:
    own = """
          <ArticleId IdType="pubmed">41000001</ArticleId>
          <ArticleId IdType="doi">10.1007/s00125-026-00001-1</ArticleId>
          <ArticleId IdType="pmc">PMC12000001</ArticleId>"""
    (paper,) = _parse(_record("41000001", own))
    assert paper.pmid == "41000001"
    assert paper.title == "Semaglutide and cardiovascular outcomes"
    assert paper.abstract == "We assessed MACE.\nMACE fell."
    assert paper.authors == ["Jane Doe", "Consortium"]
    assert paper.journal == "Diabetologia"
    assert paper.year == 2026
    assert paper.mesh_terms == ["Glucagon-Like Peptide 1"]
    assert paper.source is PaperSource.PUBMED_ABSTRACT
    assert paper.full_text is None
    assert paper.doi == "10.1007/s00125-026-00001-1"
    assert paper.pmcid == "PMC12000001"


def test_ids_of_cited_papers_are_not_taken_as_the_papers_own() -> None:
    """The reference list carries DOIs and PMC IDs too, and they are other papers'."""
    own = '<ArticleId IdType="pubmed">41000002</ArticleId>'
    (paper,) = _parse(_record("41000002", own))
    assert paper.pmcid is None
    assert paper.doi is None


def test_a_record_without_a_pmid_is_skipped() -> None:
    good = _record("41000003", '<ArticleId IdType="pubmed">41000003</ArticleId>')
    bad = "<PubmedArticle><MedlineCitation><Article/></MedlineCitation></PubmedArticle>"
    assert [p.pmid for p in _parse(bad, good)] == ["41000003"]
