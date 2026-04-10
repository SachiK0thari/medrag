"""LangGraph agent loop.

State machine:
    decompose -> retrieve -> reflect -> (re-retrieve | answer) -> end

State carries the question, sub-queries, accumulated retrieved chunks,
reflection outcomes, and final Answer.
"""

from __future__ import annotations

from medrag.models import Answer

# TODO(saturday-evening): build the LangGraph StateGraph.
# Nodes:
#   decompose_node: LLM -> list[SubQuery]
#   retrieve_node: HybridRetriever + Reranker over each sub-query
#   reflect_node: LLM judges sufficiency; sets needs_more flag
#   answer_node: LLM produces cited Answer with structured citation extraction
# Edges: conditional from reflect -> retrieve (loop, max N) or -> answer


class MedRagAgent:
    def __init__(self) -> None:
        pass

    def answer(self, question: str) -> Answer:
        raise NotImplementedError("implement Saturday evening")
