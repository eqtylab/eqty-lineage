"""Retrieved documents are the provenance a RAG chain exists to produce.

`on_retriever_start` / `on_retriever_end` had no handler at all, so a chain whose answer came from three
specific documents recorded a model call with a prompt that mentioned them and nothing establishing where
they came from.
"""

from typing import Any

import pytest
from langchain_core.documents import Document as LCDocument
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class Fake(BaseRetriever):
    """A retriever whose results are fixed, so the assets are predictable."""

    # a pydantic field, not a class attribute -- default_factory keeps instances independent
    docs: list[Any] = Field(default_factory=list)

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.docs


def _rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([("system", "answer"), ("human", "{q} ctx={ctx}")])
    model = GenericFakeChatModel(messages=iter([AIMessage("42")]))
    return {"q": lambda x: x["q"], "ctx": (lambda x: x["q"]) | retriever} | prompt | model | StrOutputParser()


def test_retrieval_is_a_computation(recording_handler):
    retriever = Fake(docs=[LCDocument(page_content="the answer is 42")])
    _rag_chain(retriever).invoke({"q": "life"}, config={"callbacks": [recording_handler]})

    kinds = [kind for _, kind, _, _ in recording_handler.computations]
    assert "retriever" in kinds, "retrieval left no trace"


def test_each_document_is_its_own_asset(recording_handler):
    docs = [LCDocument(page_content=f"doc {i}", metadata={"source": f"s{i}.txt"}) for i in range(3)]
    _rag_chain(Fake(docs=docs)).invoke({"q": "life"}, config={"callbacks": [recording_handler]})

    outputs = [outs for _, kind, _, outs in recording_handler.computations if kind == "retriever"]
    assert outputs, "no retrieval recorded"
    assert len(outputs[0]) == 3, f"expected one asset per document, got {len(outputs[0])}"


def test_identical_documents_are_one_entity(recording_handler):
    """Content-addressing should collapse a duplicate rather than register it twice."""
    doc = LCDocument(page_content="same", metadata={"source": "s.txt"})
    _rag_chain(Fake(docs=[doc, doc])).invoke({"q": "life"}, config={"callbacks": [recording_handler]})

    outputs = [outs for _, kind, _, outs in recording_handler.computations if kind == "retriever"]
    assert len(outputs[0]) == 1, "the same document twice is one entity"


def test_query_and_retriever_are_inputs(recording_handler):
    retriever = Fake(docs=[LCDocument(page_content="x")])
    _rag_chain(retriever).invoke({"q": "life"}, config={"callbacks": [recording_handler]})

    retrievals = [(ins, outs) for _, kind, ins, outs in recording_handler.computations if kind == "retriever"]
    ins, outs = retrievals[0]
    assert len(ins) >= 2, "a retrieval derives from at least the retriever and its query"
    assert outs, "a retrieval must produce something"


def test_failing_retriever_is_recorded(recording_handler):
    class Broken(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):
            raise RuntimeError("index unavailable")

    with pytest.raises(RuntimeError):
        Broken().invoke("life", config={"callbacks": [recording_handler]})

    kinds = [kind for _, kind, _, _ in recording_handler.computations]
    assert "retriever_error" in kinds, "a failed retrieval must leave a record"
