ANSWER_PROMPT = """Answer the question using only the evidence below. 
First reference conversation {history} to see if previously answers can provide context for answering the user's question.{history} can clarify references, but factual claims must be supported by the retrieved context—not by earlier assistant messages. 
Use previously retrieved passages in the conversation {history} to answer the question if it is a follow-up and previously retrieved passages can be used to provide an accurate and relevant answer to the question. 
If the previously retrieved passages in the coversation {history} are not contextually relevant to the new quetsion or only contain part of the answer or do not completely answer the user's question then retrieve more passages to provide a complete and relevant response. 
Provide at least three contextually non-overlapping examples to illustrate answer using only the evidence.
If the evidence does not answer the question, say so.
Cite claims using the source labels, and explain relevence.

Question:
{question}

Evidence:
{context}

Answer:"""

RELEVANCE_PROMPT = """Does this passage help answer the question?
Reply with only YES or NO.

Question: {question}

Passage:
{passage}"""

SUPPORT_PROMPT = """Is the answer supported by the provided evidence?
Reply with only YES or NO.

Evidence:
{context}

Answer:
{answer}"""

CONTEXTUALIZE_PROMPT = """Rewrite the current user question as a standalone search query.
Use the conversation history only to resolve references such as "she", "that", or "it".
Do not answer the question. If it is already standalone, keep its meaning unchanged.

Conversation history:
{history}

Current question: {question}

Standalone search query:"""

APPROXIMATE_PROMPT= """Rewrite the current user question so that it retains key words but better aligns with the 
available data

"""

EVALUATE_PROMPT="""You are evaluating a RAG system. The query is the user's question. Treat the retrieved contexts as the only
evidence for faithfulness. Treat the reference answer, when present, as the
source of expected facts for recall and correctness.

Return ONLY valid JSON with this schema:
{
  "context_relevance": 0.0,
  "context_precision": 0.0,
  "faithfulness_score": 0.0,
  "supported": False,
  "faithfulness_reason": "short reason or unavailable"
  "evaluation_error": "short reason or unavailable"
}

Score each metric from 0.0 to 1.0.
Judge both independently whether 1) it is topical - meaning that the answer is relevant to the query and 2) it is useful - meaning it contains informaion that helps answer the question.
context_relevance is the fraction of contexts that are topical, score 0.0 to 1.0.
context_precision is the fraction of contexts that are useful, score 0.0 to 1.0.
faithfulness_score measures whether the answer's factual claims are supported by
the contexts from 0.0 to 1.0; unsupported claims lower the score.
supported is a boolean that defines whether the faithfulness score is above the minimum faithfulness threshold.
faithfulness_reason is an explanation for why the model demonstrates the specific level of faithfulness (as defined by the faithfulness_score metric) but in plain words.
If there was an evaluation error that occured during the evaluation process note it as evalaution_errror.
If there are no contexts, use 0.0 for relevance,
precision, recall (when applicable), and faithfulness.
Do not infer missing facts. Do not reward an answer for being plausible if
the supplied evidence does not support it.
"""

