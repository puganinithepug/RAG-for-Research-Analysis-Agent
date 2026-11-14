# tests/test_grade.py
from types import SimpleNamespace
from unittest.mock import Mock

from nodes import grade


class FakeDocument:
    def __init__(self, page_content, source="manual.pdf", page=2):
        self.page_content = page_content
        self.metadata = {"source": source, "page": page}


def test_grade_one_marks_relevant_document(monkeypatch):
    document = FakeDocument("The default timeout is 30 seconds.")
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="YES")

    monkeypatch.setattr(grade, "get_grading_llm", lambda: llm)

    result = grade.grade_one({
        "question": "What is the default timeout?",
        "document": document,
        "index": 0,
        "request_id": "request-1",
        "round_id": 0,
    })

    assert result["grade_results"] == [{
        "request_id": "request-1",
        "round_id": 0,
        "index": 0,
        "status": "ok",
        "relevant": True,
    }]
    llm.invoke.assert_called_once()


def test_grade_one_records_llm_error(monkeypatch):
    document = FakeDocument("The default timeout is 30 seconds.")
    llm = Mock()
    llm.invoke.side_effect = RuntimeError("LLM unavailable")

    monkeypatch.setattr(grade, "get_grading_llm", lambda: llm)

    result = grade.grade_one({
        "question": "What is the default timeout?",
        "document": document,
        "index": 0,
        "request_id": "request-1",
        "round_id": 0,
    })

    assert result["grade_results"] == [{
        "request_id": "request-1",
        "round_id": 0,
        "index": 0,
        "status": "error",
        "error": "RuntimeError",
    }]


def test_collect_grades_keeps_relevant_documents_for_current_round():
    relevant_doc = FakeDocument("The default timeout is 30 seconds.")
    irrelevant_doc = FakeDocument("The project uses Python.")

    state = {
        "request_id": "request-1",
        "attempts": 0,
        "documents": [relevant_doc, irrelevant_doc],
        "grade_results": [
            {
                "request_id": "request-1",
                "round_id": 0,
                "index": 0,
                "status": "ok",
                "relevant": True,
            },
            {
                "request_id": "request-1",
                "round_id": 0,
                "index": 1,
                "status": "ok",
                "relevant": False,
            },
            # A result from another request should not affect this one.
            {
                "request_id": "another-request",
                "round_id": 0,
                "index": 1,
                "status": "ok",
                "relevant": True,
            },
        ],
    }

    result = grade.collect_grades(state)

    assert result["documents"] == [relevant_doc]
    assert result["grading_errors"] == []


def test_collect_grades_returns_current_round_errors():
    document = FakeDocument("The default timeout is 30 seconds.")

    result = grade.collect_grades({
        "request_id": "request-1",
        "attempts": 1,
        "documents": [document],
        "grade_results": [
            {
                "request_id": "request-1",
                "round_id": 1,
                "index": 0,
                "status": "error",
                "error": "RuntimeError",
            },
        ],
    })

    assert result["documents"] == []
    assert len(result["grading_errors"]) == 1
    assert result["grading_errors"][0]["error"] == "RuntimeError"

