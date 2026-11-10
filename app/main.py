import argparse
import os
from pathlib import Path

from app.config import settings
from app.graph import build_graph
from app.ingest import ingest
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import uuid

def initial_state(question: str) -> dict:
    return {
        "question": question,
        "request_id": uuid.uuid4().hex,
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "answer": "",
        "attempts": 0,
        "supported": False,
        "abstained": False,
        "evaluation_error": "",
        "faithfulness_reason": "",
    }

async def run_question(args):
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_db)
    ) as checkpointer:
        graph = build_graph(
            checkpointer=checkpointer,
        )

        return await graph.ainvoke(
            initial_state(args.question),
            config={
                "configurable": {
                    "thread_id": args.thread_id},
                    "max_concurrency": 2,
            },
        )

async def run_chat(thread_id):
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        while True:
            question = input("you> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            result = await graph.ainvoke(
                initial_state(question),
                config={"configurable": {"thread_id": thread_id},
                "max_concurrency": 2,}

            )
            print(result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Index files in the documents directory")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    
    chat_parser = subparsers.add_parser("chat", help="Start a chat")

    chat_parser.add_argument("--thread-id", required=True)
    ask_parser.add_argument("question")

    ask_parser.add_argument("--thread-id", required=True)

    args = parser.parse_args()

    if args.command == "ingest":
        count = ingest()
        print(f"Indexed {count} chunks.")
        return
    
    elif args.command=="ask":
        result = asyncio.run(run_question(args))

    elif args.command=="chat":
        print("To exit chat print exit or quit")
        result=asyncio.run(run_chat(args.thread_id))
        return

    else:
        print("Invalid command")
        return

    print(result["answer"])
    if result.get("evaluation_error"):
        print("Faithfulness evaluation failed; the answer was not verified.")
    elif "faithfulness_score" in result:
        print(f"Faithfulness score: {result['faithfulness_score']:.2f}")

        if not result.get("supported", False):
            print("The answer was not verified as supported by the retrieved evidence.")


if __name__ == "__main__":
    main()

