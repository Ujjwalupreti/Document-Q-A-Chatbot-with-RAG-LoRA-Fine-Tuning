"""
Prepare the CUAD (Contract Understanding Atticus Dataset) for instruction
fine-tuning.

Downloads the dataset via HuggingFace `datasets`, reformats each
(question, context, answer) triple into an instruction-following prompt,
and writes train/eval JSONL splits.

Usage:
    python prepare_dataset.py --train_size 5000 --eval_size 300
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset  
from huggingface_hub import hf_hub_download  

PROMPT_TEMPLATE = (
    "### Instruction: Answer the question based on the provided context.\n"
    "### Context: {context}\n"
    "### Question: {question}\n"
    "### Answer: {answer}"
)


def format_example(example: dict) -> dict:
    answer_text = ""
    answers = example.get("answers", {})
    if answers and answers.get("text"):
        answer_text = answers["text"][0]

    return {
        "text": PROMPT_TEMPLATE.format(
            context=example["context"].strip(),
            question=example["question"].strip(),
            answer=answer_text.strip(),
        ),
        "context": example["context"],
        "question": example["question"],
        "answer": answer_text,
    }


def main(train_size: int, eval_size: int, seed: int, out_dir: str) -> None:
    print("Downloading CUAD raw JSON asset via huggingface_hub...")
    
    try:
        local_json_path = hf_hub_download(
            repo_id="theatticusproject/cuad-qa",
            repo_type="dataset",
            filename="cuad-qa.json"
        )
        
        with open(local_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        dataset = Dataset.from_list(raw_data["data"])
        print(f"Successfully loaded {len(dataset)} items from local cache.")
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to securely fetch or parse the CUAD dataset: {e}. "
            "Verify your internet connection and huggingface_hub installation."
        )

    dataset = dataset.filter(lambda ex: len(ex["answers"]["text"]) > 0)
    dataset = dataset.shuffle(seed=seed)

    total_needed = train_size + eval_size
    if len(dataset) < total_needed:
        raise ValueError(
            f"Dataset only has {len(dataset)} usable examples, "
            f"but {total_needed} were requested."
        )

    subset = dataset.select(range(total_needed))
    train_examples = subset.select(range(train_size))
    eval_examples = subset.select(range(train_size, total_needed))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train_examples), ("eval", eval_examples)]:
        file_path = out_path / f"{name}.jsonl"
        with open(file_path, "w") as f:
            for ex in split:
                f.write(json.dumps(format_example(ex)) + "\n")
        print(f"Wrote {len(split)} examples to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--eval_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()
    main(args.train_size, args.eval_size, args.seed, args.out_dir)