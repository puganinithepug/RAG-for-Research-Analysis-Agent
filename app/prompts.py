ANSWER_PROMPT = """Answer the question using only the evidence below. Provide at least three examples to illustrate answer using only the evidence.
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

