# tests/test_evaluation.py
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nodes import evaluate as evaluation


class FakeDocument:
    def __init__(self, page_content: str):
        self.page_content = page_content


def fake_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]
    )


@pytest.mark.asyncio
async def test_evaluate_documentation_question_returns_metrics(monkeypatch):
    evaluator_result = {
        "context_relevance": 1.0,
        "context_precision": 1.0,
        "faithfulness": 1.0,
        "faithfulness_reason": (
            "The answer's 30-second timeout is stated in the documentation."
        ),
    }

    mock_create = AsyncMock(
        return_value=fake_response(json.dumps(evaluator_result))
    )
    monkeypatch.setattr(
        evaluation.client.chat.completions,
        "create",
        mock_create,
    )

    state = {
        "question": "What is the default request timeout?",
        "answer": "The default request timeout is 30 seconds.",
        "documents": [
            FakeDocument(
                "HTTP Client Configuration\n"
                "The request_timeout setting controls how long the client "
                "waits for a response. Its default value is 30 seconds."
            ),
            FakeDocument(
                "Retries\n"
                "The client retries failed requests up to three times."
            ),
        ],
    }

    result = await evaluation.evaluate(state)

    assert result["context_relevance"] == 1.0
    assert result["context_precision"] == 1.0
    assert result["faithfulness_score"] == 1.0
    assert result["supported"] is True
    assert "30-second timeout" in result["faithfulness_reason"]
    assert result["evaluation_error"] == ""

    # Confirm the documentation question, answer, and retrieved contexts
    # were sent to the evaluator.
    call = mock_create.await_args
    payload = json.loads(call.kwargs["messages"][1]["content"])

    assert payload["question"] == state["question"]
    assert payload["answer"] == state["answer"]
    assert any(
        "default value is 30 seconds" in context
        for context in payload["contexts"]
    )


@pytest.mark.asyncio
async def test_evaluate_no_documents_returns_zero_scores():
    result = await evaluation.evaluate({
        "question": "What is the default request timeout?",
        "answer": "The default request timeout is 30 seconds.",
        "documents": [],
    })

    assert result["context_relevance"] == 0.0
    assert result["context_precision"] == 0.0
    assert result["faithfulness_score"] == 0.0
    assert result["supported"] is False
    assert result["evaluation_error"] == ""

