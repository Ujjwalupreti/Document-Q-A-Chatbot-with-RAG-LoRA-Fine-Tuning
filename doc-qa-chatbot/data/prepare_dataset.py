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
    "### Instruction:\n"
    "You are an intelligent document assistant. "
    "Answer the user's question ONLY using the provided document.\n\n"
    "### Document:\n"
    "{context}\n\n"
    "### Question:\n"
    "{question}\n\n"
    "### Response:\n"
    "{answer}"
)

def _normalize_example(example):

    context = example.get("context", "").strip()
    question = example.get("question", "").strip()
    answers = example.get("answers", {})

    if isinstance(answers, dict):
        answer_list = answers.get("text", [])
        answer = answer_list[0].strip() if len(answer_list) > 0 else ""
    else:
        answer = ""
    if not context:
        return None
    if not question:
        return None
    if answer == "":
        return None

    return {
        "text": PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            answer=answer,
        ),
        "context": context,
        "question": question,
        "answer": answer,
    }


def _load_squad():

    from datasets import load_dataset

    print("Loading SQuAD v2...")

    dataset = load_dataset(
        "squad_v2",
        split="train",
        token=os.environ.get("HF_TOKEN", None),
    )

    print(f"Loaded {len(dataset):,} examples")

    return dataset


def main(train_size: int,eval_size: int,seed: int,out_dir: str,force: bool = False,) -> None:

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.jsonl"
    eval_file = out_path / "eval.jsonl"

    if train_file.exists() and eval_file.exists() and not force:
        print("Dataset already exists.")
        print(f"Train : {train_file}")
        print(f"Eval  : {eval_file}")
        print("Use --force to regenerate the dataset.")
        return

    hf_token = os.environ.get("HF_TOKEN", "").strip()

    if hf_token:
        try:
            from huggingface_hub import login
            login(
                token=hf_token,
                add_to_git_credential=False,
            )
            print("Logged in to Hugging Face.")
        except Exception as exc:
            print(f"Warning: HF login failed: {exc}")

    raw_ds = _load_squad()
    examples = []

    for row in raw_ds:
        normed = _normalize_example(row)
        if normed is not None:
            examples.append(normed)

    print(f"Valid examples: {len(examples):,}")

    if len(examples) == 0:
        raise RuntimeError("No usable examples were found in the dataset.")

    total_needed = train_size + eval_size

    if len(examples) < total_needed:
        print(f"Requested {total_needed} examples but only {len(examples)} are available.")
        
        train_size = int(len(examples) * 0.9)
        eval_size = len(examples) - train_size
        
        print(f"Automatically using {train_size} train / {eval_size} eval examples.")

    import random

    rng = random.Random(seed)
    rng.shuffle(examples)

    train_examples = examples[:train_size]
    eval_examples = examples[train_size : train_size + eval_size]

    for filename, split in [
        ("train.jsonl", train_examples),
        ("eval.jsonl", eval_examples),
    ]:
        file_path = out_path / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            for example in split:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
                
        print(f"Wrote {len(split):,} examples -> {file_path}")

    stats = {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "seed": seed,
    }

    with open(out_path / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Dataset preparation completed successfully.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare CUAD dataset for LoRA fine-tuning."
    )
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--eval_size",  type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out_dir",    type=str, default=".")
    parser.add_argument("--force",action="store_true",help="Regenerate train/eval JSONL even if already present.")
    args = parser.parse_args()
    main(args.train_size, args.eval_size, args.seed, args.out_dir,args.force,)