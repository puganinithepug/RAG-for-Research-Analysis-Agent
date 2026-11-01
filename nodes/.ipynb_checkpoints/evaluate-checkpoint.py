from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from app.config import settings
from app.state import RAGState

client = AsyncOpenAI(
    api_key="ollama",  # Ollama does not require a real API key here
    base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
)

evaluator_llm = llm_factory(
    settings.evaluator_model,
    provider="openai",
    client=client,
)

faithfulness_metric = Faithfulness(llm=evaluator_llm)

async def evaluate(state: RAGState) -> dict:
    documents = state.get("documents", [])
    answer = state.get("answer", "")

    # No evidence means there is nothing for RAGAS to verify.
    if not documents:
        return {
            "faithfulness_score": 0.0,
            "supported": False,
            "faithfulness_reason": "No retrieved evidence to evaluate.",
            "evaluation_error": "",
        }

    contexts = [document.page_content for document in documents]

    try:
        result = await faithfulness_metric.ascore(
            user_input=state["question"],
            response=answer,
            retrieved_contexts=contexts,
        )

        score = float(result.value)
        return {
            "faithfulness_score": score,
            "supported": score >= settings.faithfulness_threshold,
            "faithfulness_reason": getattr(result, "reason", "") or "",
            "evaluation_error": "",
        }

    except Exception as exc:
        # Don't turn an evaluator outage into a fake faithfulness score.
        return {
            "supported": False,
            "evaluation_error": str(exc),
        }

