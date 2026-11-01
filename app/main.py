import argparse

from app.config import settings
from app.graph import build_graph
from app.ingest import ingest
import asyncio

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Index files in the documents directory")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")

    args = parser.parse_args()

    if args.command == "ingest":
        count = ingest()
        print(f"Indexed {count} chunks.")
        return

    result = asyncio.run(build_graph().ainvoke({
        "question": args.question,
        "documents": [],
        "answer": "",
        "attempts": 0,
        "supported": False,
    }))
    print(result["answer"])
    if result.get("evaluation_error"):
        print("Faithfulness evaluation failed; the answer was not verified.")
    elif "faithfulness_score" in result:
        print(f"Faithfulness score: {result['faithfulness_score']:.2f}")

        if not result.get("supported", False):
            print("The answer was not verified as supported by the retrieved evidence.")


if __name__ == "__main__":
    main()

