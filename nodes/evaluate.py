import json

from openai import AsyncOpenAI

from app.config import settings
from app.state import RAGState

client = AsyncOpenAI(
    api_key="ollama",
    base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
)


async def evaluate(state: RAGState) -> dict:
    documents = state.get("documents", [])
    answer = state.get("answer", "")

    if not documents:
        return {
            "faithfulness_score": 0.0,
            "supported": False,
            "faithfulness_reason": "No retrieved evidence to evaluate.",
            "evaluation_error": "",
        }

    contexts = [document.page_content for document in documents]

    try:
        response = await client.chat.completions.create(
            model=settings.evaluator_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate whether the answer's factual claims are supported "
                        "by the supplied context. Do not use outside knowledge. "
                        "Return only a JSON object with keys: score (number from 0 "
                        "to 1) and reason (short string). A score of 1 means all "
                        "factual claims are supported; 0 means none are supported."
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

        raw = (response.choices[0].message.content or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end < start:
            raise ValueError(f"Evaluator did not return JSON: {raw}")

        result = json.loads(raw[start : end + 1])
        score = float(result["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Evaluator score out of range: {score}")

        return {
            "faithfulness_score": score,
            "supported": score >= settings.faithfulness_threshold,
            "faithfulness_reason": str(result.get("reason", "")),
            "evaluation_error": "",
        }

    except Exception as exc:
        return {
            "supported": False,
            "evaluation_error": str(exc),
        }

