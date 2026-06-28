"""
LoRA / QLoRA fine-tuning script for Llama 3.2 (3B).

    two-layer guard applied at the TOP of this file, before any
    transformers/trl import:

      Layer 1 (env var): set KERAS_BACKEND=torch  →  tells Keras 3 to use
        the PyTorch backend, which is compatible with the rest of the stack.

      Layer 2 (env var): set TF_CPP_MIN_LOG_LEVEL=3  →  suppresses TF noise.

      Layer 3 (package): if the env-var fix alone is not enough for your
        environment, install tf-keras instead of keras (see
        requirements_train.txt).  The env var approach is tried first because
        it requires zero package changes.

Usage:
    python train_lora.py \\
        --base_model meta-llama/Llama-3.2-3B-Instruct \\
        --train_file ../data/train.jsonl \\
        --output_dir ./lora-doc-qa \\
        --epochs 3
"""

# ---------------------------------------------------------------------------
# Bug-3 guard — must run BEFORE any transformers / trl import.
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("KERAS_BACKEND", "torch")   # Keras 3 → use torch backend
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress TF C++ log spam
# Prevent transformers from ever touching the Keras import path when we are
# running a pure-PyTorch training loop.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
# ---------------------------------------------------------------------------

import argparse
import logging

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _check_keras_compat() -> None:
    """
    Emit a clear warning if Keras 3 is present and the env-var fix may not
    be enough, so the user knows exactly what to downgrade.
    """
    try:
        import keras  # noqa: PLC0415
        major = int(keras.__version__.split(".")[0])
        if major >= 3:
            backend = getattr(keras, "backend", lambda: "unknown")()
            if callable(backend):
                backend = backend()
            logger.warning(
                "Keras %s detected (backend: %s). "
                "If training fails with 'Keras 3 not yet supported', run:\n"
                "  pip install tf-keras --upgrade\n"
                "  pip install 'keras<3' --upgrade\n"
                "See requirements_train.txt for the pinned safe versions.",
                keras.__version__,
                backend,
            )
    except ImportError:
        pass  # Keras not installed at all — no problem.


_check_keras_compat()

# ---------------------------------------------------------------------------
# Now it is safe to import TRL / PEFT / transformers.
# ---------------------------------------------------------------------------
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer  # noqa: E402


def build_model_and_tokenizer(base_model: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    model, tokenizer = build_model_and_tokenizer(args.base_model)
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")

    eval_dataset   = None
    eval_strategy  = "no"
    if args.eval_file and os.path.exists(args.eval_file):
        try:
            eval_dataset  = load_dataset("json", data_files=args.eval_file, split="train")
            eval_strategy = "epoch"
            logger.info("Loaded %d eval examples for periodic evaluation.", len(eval_dataset))
        except Exception as exc:
            logger.warning("Could not load eval file '%s', training without eval: %s", args.eval_file, exc)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy=eval_strategy,
        bf16=True,
        report_to="none",
        # FIX: max_seq_length belongs on SFTConfig in trl >= 0.12, not as a
        # separate kwarg to SFTTrainer.  Pinning it here keeps the script
        # compatible with both old and new trl versions.
        max_seq_length=2048,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("LoRA adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Llama 3.2 on the prepared CUAD dataset."
    )
    parser.add_argument("--base_model",  type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train_file",  type=str, default="../data/train.jsonl")
    parser.add_argument("--eval_file",   type=str, default="../data/eval.jsonl")
    parser.add_argument("--output_dir",  type=str, default="./lora-doc-qa")
    parser.add_argument("--epochs",      type=int, default=3)
    args = parser.parse_args()
    main(args)