"""
Evaluate the LoRA fine-tuned model against the zero-shot base model on the
held-out eval set, using SQuAD-style Exact Match / F1 and ROUGE-L.

This is what produces the accuracy / improvement-over-baseline numbers
referenced on the resume. Run it once before fine-tuning is applied (or
pass --skip_base) and once after to get both sides of the comparison.

Usage:
    python evaluate.py \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --adapter_dir ./lora-doc-qa \
        --eval_file ../data/eval.jsonl
"""
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT_TEMPLATE = (
    "### Instruction: Answer the question based on the provided context.\n"
    "### Context: {context}\n"
    "### Question: {question}\n"
    "### Answer:"
)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def load_model(base_model: str, adapter_dir: Optional[str]):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    if adapter_dir:
        # F-03: Check that the adapter directory exists before trying to load.
        adapter_path = Path(adapter_dir)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {adapter_path.resolve()}\n"
                "Run train_lora.py first to fine-tune the model."
            )
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, context: str, question: str, max_new_tokens: int = 128) -> str:
    """F-05: Aligned max_new_tokens=128 to match the inference LLM (llm.py)."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Answer:")[-1].strip()


def run_eval(model, tokenizer, eval_file: str) -> dict:
    dataset = load_dataset("json", data_files=eval_file, split="train")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    em_total, f1_total, rouge_total = 0.0, 0.0, 0.0
    n = len(dataset)

    for example in dataset:
        prediction = generate_answer(model, tokenizer, example["context"], example["question"])
        ground_truth = example["answer"]

        em_total += exact_match_score(prediction, ground_truth)
        f1_total += f1_score(prediction, ground_truth)
        rouge_total += scorer.score(ground_truth, prediction)["rougeL"].fmeasure

    return {
        "exact_match": em_total / n,
        "f1": f1_total / n,
        "rougeL": rouge_total / n,
        "n_examples": n,
    }


def main(args: argparse.Namespace) -> None:
    results = {}

    if not args.skip_base:
        print("Evaluating zero-shot base model...")
        base_model, tokenizer = load_model(args.base_model, adapter_dir=None)
        results["zero_shot"] = run_eval(base_model, tokenizer, args.eval_file)
        del base_model
        torch.cuda.empty_cache()

    print("Evaluating LoRA fine-tuned model...")
    tuned_model, tokenizer = load_model(args.base_model, adapter_dir=args.adapter_dir)
    results["fine_tuned"] = run_eval(tuned_model, tokenizer, args.eval_file)

    print(f"\n=== Results on {results['fine_tuned']['n_examples']} eval examples ===")
    header = f"{'Metric':<15}"
    if "zero_shot" in results:
        header += f"{'Zero-shot':<15}{'Fine-tuned':<15}{'Delta':<10}"
    else:
        header += f"{'Fine-tuned':<15}"
    print(header)

    for metric in ["exact_match", "f1", "rougeL"]:
        tuned_val = results["fine_tuned"][metric]
        if "zero_shot" in results:
            base_val = results["zero_shot"][metric]
            delta = tuned_val - base_val
            print(f"{metric:<15}{base_val:<15.3f}{tuned_val:<15.3f}{delta:+.3f}")
        else:
            print(f"{metric:<15}{tuned_val:<15.3f}")

    # F-04: Write to the same directory as this script (finetune/) so the
    # API's /metrics endpoint can always find it, regardless of the CWD.
    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="./lora-doc-qa")
    parser.add_argument("--eval_file", type=str, default="../data/eval.jsonl")
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip the zero-shot baseline run (e.g. if you already have those results saved).",
    )
    args = parser.parse_args()
    main(args)
