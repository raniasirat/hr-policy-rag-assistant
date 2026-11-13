"""Interactive command-line HR assistant over the indexed documents.

Usage:
    python scripts/chat.py
Type a question and press Enter. Type 'exit' or 'quit' (or press Ctrl+C) to leave.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigurationError, configure_logging, load_settings  # noqa: E402
from src.rag_pipeline import AnswerResult, RAGPipeline, RAGPipelineError  # noqa: E402

EXIT_COMMANDS = {"exit", "quit", "q", ":q"}


def render(result: AnswerResult) -> str:
    """Format an answer and its sources for the terminal."""
    lines = ["", result.answer]
    sources = result.format_sources()
    if sources:
        lines += ["", "Sources:", sources]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with your HR documents.")
    parser.add_argument("--top-k", type=int, default=None, help="override TOP_K for this session")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    pipeline = RAGPipeline.from_settings(settings)
    print("HR Assistant. Ask a question about the HR policies and employee records.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            return 0
        try:
            result = pipeline.ask(question, top_k=args.top_k)
        except RAGPipelineError as exc:
            print(f"\n[error] {exc}\n")
            continue
        print(render(result))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
