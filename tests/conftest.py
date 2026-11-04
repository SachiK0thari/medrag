"""Shared pytest fixtures."""

from __future__ import annotations

import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")

import pytest

from medrag.models import Paper, PaperSource


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        pmid="12345678",
        pmcid="PMC1234567",
        title="Liraglutide and Cardiovascular Outcomes in Type 2 Diabetes",
        abstract=(
            "Background: The cardiovascular safety of liraglutide is unknown. "
            "Methods: We randomly assigned patients with type 2 diabetes and high "
            "cardiovascular risk to liraglutide or placebo. Results: The primary "
            "composite outcome occurred in significantly fewer patients in the "
            "liraglutide group than in the placebo group."
        ),
        full_text=None,
        authors=["John Smith", "Jane Doe"],
        journal="N Engl J Med",
        year=2016,
        doi="10.1056/NEJMoa1603827",
        mesh_terms=["Liraglutide", "Cardiovascular Diseases"],
        source=PaperSource.PUBMED_ABSTRACT,
        url="https://pubmed.ncbi.nlm.nih.gov/12345678/",  # type: ignore[arg-type]
    )


@pytest.fixture
def sample_full_text_paper(sample_paper: Paper) -> Paper:
    long_body = (
        "Introduction\n" + ("Background context. " * 50) + "\n\n"
        "Methods\n" + ("We randomized patients. " * 50) + "\n\n"
        "Results\n" + ("MACE occurred in 13.0% vs 14.9%. " * 50) + "\n\n"
        "Discussion\n" + ("These findings suggest. " * 50)
    )
    return sample_paper.model_copy(update={"full_text": long_body})
