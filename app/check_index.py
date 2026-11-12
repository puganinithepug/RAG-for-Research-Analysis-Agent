from collections import Counter

from app.retrieval import get_vectorstore

vs = get_vectorstore()
data = vs.get(include=["metadatas"])

print("Indexed chunks:", len(data["ids"]))
print(Counter(
    m.get("source", "<missing source>")
    for m in data["metadatas"]
))

