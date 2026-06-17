"""
Singleton loader for the LoRA fine-tuned Llama 3.2 model, so it is loaded
once at API startup and reused across all requests (avoids reload overhead
on every chat call).
"""
import logging
import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "finetune/lora-doc-qa"

PROMPT_TEMPLATE = (
    "### Instruction: Answer the question based on the provided context.\n"
    "### Context: {context}\n"
    "### Question: {question}\n"
    "### Answer:"
)


class LLM:
    _model = None
    _tokenizer = None
    _adapter_loaded = False

    @classmethod
    def load(cls) -> None:
        if cls._model is not None:
            return

        quantization_config = None
        device_map = "auto"

        if torch.cuda.is_available() and platform.system() == "Linux":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("Using 4-bit quantization (bitsandbytes).")
            except ImportError:
                logger.warning("bitsandbytes not installed, loading without quantization.")
        elif torch.cuda.is_available():
            logger.info("CUDA available but not Linux — loading in float16 without bitsandbytes.")
        else:
            logger.info("No CUDA available — loading model on CPU (this will be slow).")
            device_map = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {"device_map": device_map}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        elif torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)

        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER_DIR)
            cls._adapter_loaded = True
            logger.info("Loaded LoRA adapter from %s", ADAPTER_DIR)
        except Exception as exc:
            cls._adapter_loaded = False
            logger.warning(
                "No LoRA adapter loaded from %s — using base model. Reason: %s",
                ADAPTER_DIR, exc,
            )

        model.eval()
        cls._model = model
        cls._tokenizer = tokenizer

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def is_adapter_loaded(cls) -> bool:
        """Whether the LoRA adapter was successfully loaded on top of the base model."""
        return cls._adapter_loaded

    @classmethod
    @torch.no_grad()
    def generate(cls, context: str, question: str, max_new_tokens: int = 128) -> str:
        if cls._model is None:
            raise RuntimeError("Model not loaded. Call LLM.load() at startup.")

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = cls._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(cls._model.device)

        output = cls._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = cls._tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded.split("### Answer:")[-1].strip()