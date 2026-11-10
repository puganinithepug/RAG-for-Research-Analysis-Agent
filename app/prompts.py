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

EVALUATE_PROMPT="""Evaluate whether the answer's factual claims are supported
by the supplied context. Do not use outside knowledge.
Return only a JSON object with keys: score (number from 0 to 1) and reason (short string). 
A score of 1 means all claims are factual claims and are supported; 0 means none are supported.
"""
