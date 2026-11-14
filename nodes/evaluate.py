import json

from openai import AsyncOpenAI

from app.config import settings
from app.state import RAGState
from app.prompts import EVALUATE_PROMPT

client = AsyncOpenAI(
    api_key="ollama",
    base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
)

MAX_DOCS = 11
MAX_CHARS_PER_DOC = 3000
MAX_TOTAL_CHARS = 33000

def shorten_contexts(documents: list) -> list[str]:
    contexts = []
    total = 0

    for i, doc in enumerate(documents[:MAX_DOCS], start=1):
        text = doc.page_content.strip()
        remaining = MAX_TOTAL_CHARS - total
        if remaining <= 0:
            break

        text = text[:min(MAX_CHARS_PER_DOC, remaining)]
        contexts.append(f"[Document {i}]\n{text}")
        total += len(text)

    return contexts


import time

async def evaluate(state: RAGState) -> dict:
    documents = state.get("documents", [])
    answer = state.get("answer", "")

    if not documents:
        output = {
            "context_relevance": 0.0,
            "context_precision": 0.0,
            "faithfulness_score": 0.0,
            "supported": False,
            "faithfulness_reason": "No retrieved evidence to evaluate.",
            "evaluation_error": "",
        }
        print("[evaluate] Skipping: no documents", flush=True)
        return output

    contexts = shorten_contexts(documents)
    print(
        f"[evaluate] Using {len(contexts)} contexts, "
        f"{sum(map(len, contexts))} total characters",
        flush=True,
    )
    context_chars = sum(len(context) for context in contexts)

    print(
        f"[evaluate] Starting: model={settings.evaluator_model}, "
        f"documents={len(contexts)}, context_chars={context_chars}",
        flush=True,
    )

    started = time.perf_counter()

    try:
        print("[evaluate] Waiting for Ollama response...", flush=True)
        response = await client.chat.completions.create(
            model=settings.evaluator_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        EVALUATE_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "question": state.get("question", ""),
                        "answer": answer,
                        "contexts": contexts,
                    }),
                },
            ],
        )

        elapsed = time.perf_counter() - started
        print(f"[evaluate] Ollama responded after {elapsed:.2f}s", flush=True)

        raw = (response.choices[0].message.content or "").strip()
        print(f"[evaluate] Response length: {len(raw)} characters", flush=True)

        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end < start:
            raise ValueError(f"Evaluator did not return JSON: {raw}")

        result = json.loads(raw[start : end + 1])
        
        def get_score(name: str) -> float | None:
            value = result.get(name)
            if value is None:
                return None
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} out of range: {value}")
            return value
        
        faithfulness = get_score("faithfulness")

        output =  {
            "context_relevance": get_score("context_relevance"),
            "context_precision": get_score("context_precision"),
            "faithfulness_score": faithfulness,
            "supported": faithfulness is not None and faithfulness >= settings.faithfulness_threshold,
            "faithfulness_reason": str(result.get("faithfulness", "")),
            "evaluation_error": "",
        }
        print(output)
        return output

    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"[evaluate] Failed after {elapsed:.2f}s: {exc}", flush=True)
        return {
            "context_relevance": 0.0,
            "context_precision": 0.0,
            "faithfulness_score": 0.0,
            "supported": False,
            "faithfulness_reason": "Evaluation failed.",
            "evaluation_error": str(exc),
        }

