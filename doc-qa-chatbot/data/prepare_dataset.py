"""
Prepare the CUAD (Contract Understanding Atticus Dataset) for instruction
fine-tuning.

Downloads the dataset, reformats each (question, context, answer) triple into
an instruction-following prompt, and writes train/eval JSONL splits.

Usage:
    python prepare_dataset.py --train_size 5000 --eval_size 300

Requirements:
    pip install datasets huggingface_hub
    huggingface-cli login  (or set HF_TOKEN env var)
"""

import argparse
import json
import os
from pathlib import Path

PROMPT_TEMPLATE = (
    "### Instruction: Answer the question based on the provided context.\n"
    "### Context: {context}\n"
    "### Question: {question}\n"
    "### Answer: {answer}"
)

_DATASET_CANDIDATES = [
    ("theatticusproject/cuad-qa", "train"),
    ("theatticusproject/cuad-qa", "validation"),
    ("cuad", "train"),
]


def _normalize_example(example: dict) -> dict | None:
    """
    Normalise field names across the two known CUAD schema variants:

    Variant A (theatticusproject/cuad-qa):
        context, question, answers.text[]

    Variant B (cuad):
        context, question, answers.text[]   ← same schema, just different repo

    Returns None if the example has no extractable answer (skip it).
    """
    context  = (example.get("context") or "").strip()
    question = (example.get("question") or "").strip()

    answers = example.get("answers", {}) or {}
    texts   = answers.get("text", []) or []
    answer  = texts[0].strip() if texts else ""

    if not context or not question or not answer:
        return None

    return {
        "text": PROMPT_TEMPLATE.format(
            context=context, question=question, answer=answer
        ),
        "context":  context,
        "question": question,
        "answer":   answer,
    }


def _load_cuad_dataset():
    """
    Try each candidate in _DATASET_CANDIDATES and return the first that
    loads successfully.  Raises RuntimeError if all candidates fail.
    """
    from datasets import load_dataset  

    last_error: Exception | None = None
    for repo_id, split_name in _DATASET_CANDIDATES:
        try:
            print(f"Trying  load_dataset('{repo_id}', split='{split_name}') …")
            ds = load_dataset(repo_id, split=split_name, trust_remote_code=False)
            print(f"  → loaded {len(ds):,} rows from '{repo_id}' split='{split_name}'")
            return ds
        except Exception as exc:  
            print(f"  → failed: {exc}")
            last_error = exc

    raise RuntimeError(
        f"Last error: {last_error}"
    )


def main(train_size: int, eval_size: int, seed: int, out_dir: str) -> None:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        try:
            from huggingface_hub import login  
            login(token=hf_token, add_to_git_credential=False)
            print("Logged in to HuggingFace Hub via HF_TOKEN.")
        except Exception as exc:  
            print(f"Warning: could not log in with HF_TOKEN: {exc}")

    raw_ds = _load_cuad_dataset()
    examples = []
    for row in raw_ds:
        normed = _normalize_example(row)
        if normed is not None:
            examples.append(normed)

    print(f"Answerable examples after filtering: {len(examples):,}")

    total_needed = train_size + eval_size
    if len(examples) < total_needed:
        raise ValueError(
            f"Only {len(examples):,} usable examples available, "
            f"but {total_needed:,} were requested "
            f"(--train_size {train_size} + --eval_size {eval_size}).\n"
            "Lower --train_size / --eval_size or use the full dataset."
        )

    import random  
    rng = random.Random(seed)
    rng.shuffle(examples)

    train_examples = examples[:train_size]
    eval_examples  = examples[train_size : train_size + eval_size]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train_examples), ("eval", eval_examples)]:
        file_path = out_path / f"{name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split):,} examples → {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare CUAD dataset for LoRA fine-tuning."
    )
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--eval_size",  type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out_dir",    type=str, default=".")
    args = parser.parse_args()
    main(args.train_size, args.eval_size, args.seed, args.out_dir)