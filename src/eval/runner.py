import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from src.llm.qa import answer_question

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    id: str
    question: str
    expected_substrings: List[str]
    vector_store_path: str = "data/vector_store"
    top_k: int = 5


def load_dataset(path: Path) -> List[EvalSample]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    samples: List[EvalSample] = []
    for item in raw:
        samples.append(
            EvalSample(
                id=item["id"],
                question=item["question"],
                expected_substrings=item.get("expected_substrings", []),
                vector_store_path=item.get("vector_store_path", "data/vector_store"),
                top_k=item.get("top_k", 5),
            )
        )
    return samples


def evaluate_sample(sample: EvalSample) -> dict:
    response = answer_question(
        question=sample.question,
        vector_store_path=sample.vector_store_path,
        n_results=sample.top_k,
    )
    answer = response.get("answer", "")
    matched = [sub for sub in sample.expected_substrings if sub.lower() in answer.lower()]
    success = len(matched) == len(sample.expected_substrings)
    return {
        "id": sample.id,
        "question": sample.question,
        "success": success,
        "matched": matched,
        "missing": [
            sub for sub in sample.expected_substrings if sub not in matched
        ],
        "answer": answer,
        "sources": response.get("sources", []),
    }


def evaluate_dataset(dataset_path: str) -> dict:
    dataset = load_dataset(Path(dataset_path))
    results = [evaluate_sample(sample) for sample in dataset]
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    summary = {
        "dataset": dataset_path,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "results": results,
    }
    return summary


def _format_summary(summary: dict) -> str:
    lines = [
        f"Dataset: {summary['dataset']}",
        f"Passed: {summary['passed']} / {summary['total']}"
    ]
    for result in summary["results"]:
        status = "✅" if result["success"] else "⚠️"
        missing = ", ".join(result["missing"]) if result["missing"] else ""
        lines.append(f"{status} {result['id']} - missing: {missing}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA samples against the knowledge base")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to a JSON dataset with samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO)
    summary = evaluate_dataset(args.dataset)
    print(_format_summary(summary))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Wrote evaluation summary to %s", output_path)


if __name__ == "__main__":
    main()
