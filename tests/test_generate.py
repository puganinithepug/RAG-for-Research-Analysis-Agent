# tests/test_generate.py
from types import SimpleNamespace
from unittest.mock import Mock

from nodes import generate


class FakeDocument:
    def __init__(self, page_content, source="manual.pdf", page=2):
        self.page_content = page_content
        self.metadata = {"source": source, "page": page}


def test_generate_returns_fallback_when_no_documents():
    result = generate.generate({
        "question": "What is the default timeout?",
        "documents": [],
        "messages": [],
    })

    assert result["answer"] == (
        "I couldn't find relevant evidence in the indexed documents."
    )
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == result["answer"]


def test_generate_uses_context_and_adds_cited_source(monkeypatch):
    document = FakeDocument("The default timeout is 30 seconds.")

    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content="The default timeout is 30 seconds [1]."
    )

    monkeypatch.setattr(generate, "get_llm", lambda **kwargs: llm)
    monkeypatch.setattr(generate, "temp", lambda question: 0)

    result = generate.generate({
        "question": "What is the default timeout?",
        "documents": [document],
        "messages": [],
    })

    assert result["answer"] == (
        "The default timeout is 30 seconds [1].\n\n"
        "Sources:\n[1] manual.pdf, page 2"
    )
    assert result["messages"][0].content == result["answer"]

    prompt = llm.invoke.call_args.args[0]
    assert "What is the default timeout?" in prompt
    assert "The default timeout is 30 seconds." in prompt


def test_generate_does_not_add_source_for_uncited_document(monkeypatch):
    document = FakeDocument("The default timeout is 30 seconds.")

    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content="The documentation says the timeout is 30 seconds."
    )

    monkeypatch.setattr(generate, "get_llm", lambda **kwargs: llm)
    monkeypatch.setattr(generate, "temp", lambda question: 0)

    result = generate.generate({
        "question": "What is the default timeout?",
        "documents": [document],
        "messages": [],
    })

    assert result["answer"] == (
        "The documentation says the timeout is 30 seconds."
    )

