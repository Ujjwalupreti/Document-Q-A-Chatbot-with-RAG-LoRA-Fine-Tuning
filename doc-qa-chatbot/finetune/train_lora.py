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