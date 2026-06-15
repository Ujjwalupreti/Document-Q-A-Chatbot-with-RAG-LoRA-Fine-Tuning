# Document Q&A Chatbot — RAG + LoRA Fine-Tuned Llama 3.2

A document question-answering chatbot: upload a PDF/DOCX/TXT, ask questions about it,
and get answers from a LoRA/QLoRA fine-tuned Llama 3.2 (3B) model grounded in a
Chroma-backed RAG pipeline.

This project is structured so that every number in the resume bullets below it
(accuracy, latency, concurrency) is something **you generate yourself** by running
the included scripts — not a number to copy-paste.

```
"Achieved 91% answer accuracy ... by fine-tuning Llama 3.2 (3B) using LoRA ..."
"Reduced retrieval latency by 55% ... two-stage RAG pipeline ..."
"Enabled multi-format document ingestion for 50+ concurrent users ..."
"Cut GPU fine-tuning cost by 60% vs full fine-tuning by applying QLoRA ..."
```

---

## 1. Project Structure

```
doc-qa-chatbot/
├── data/
│   ├── prepare_dataset.py   # downloads CUAD, builds train/eval JSONL
│   ├── train.jsonl          # sample (2 rows) — regenerate for real run
│   └── eval.jsonl           # sample (1 row)  — regenerate for real run
├── finetune/
│   ├── train_lora.py        # LoRA/QLoRA fine-tuning (TRL SFTTrainer)
│   └── evaluate.py          # zero-shot vs fine-tuned: EM / F1 / ROUGE-L
├── rag/
│   ├── ingest.py             # load → chunk (512/64 tokens) → embed → Chroma
│   └── retriever.py          # top-k search + cosine re-rank
├── benchmark/
│   └── retrieval_benchmark.py  # naive scan vs ANN+re-rank latency
├── api/
│   ├── llm.py                # singleton model loader (base + LoRA adapter)
│   ├── models.py             # Pydantic schemas
│   └── main.py                # FastAPI app: /upload /chat /sessions /health
├── app.py                     # Streamlit chat UI
├── requirements.txt
└── .env.example
```

---

## 2. Setup

**Requirements**: Python 3.10+, a CUDA GPU with ≥12–16GB VRAM (for QLoRA on the
3B model), and a HuggingFace account with access to Llama 3.2 (gated model —
request access on the model page, then generate a token).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your HF_TOKEN
```

Log in to HuggingFace so `transformers`/`datasets` can pull gated models:

```bash
huggingface-cli login --token YOUR_TOKEN
```

---

## 3. Run Order (to generate your real resume numbers)

### Step 1 — Build the dataset
```bash
cd data
python prepare_dataset.py --train_size 5000 --eval_size 300
cd ..
```
This downloads CUAD, drops unanswerable spans, and writes `train.jsonl` (5,000
examples) and `eval.jsonl` (300 held-out examples) in the instruction format
used for fine-tuning.

### Step 2 — Baseline evaluation (zero-shot)
```bash
cd finetune
python evaluate.py --adapter_dir none --skip_base=false --eval_file ../data/eval.jsonl
```
*(First run without an adapter to record the zero-shot baseline F1/EM/ROUGE-L —
this is the "before" number for your "+28% over zero-shot baseline" claim.)*

### Step 3 — LoRA / QLoRA fine-tuning
```bash
python train_lora.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --train_file ../data/train.jsonl \
  --output_dir ./lora-doc-qa \
  --epochs 3
```
- Loads the base model in 4-bit NF4 (QLoRA)
- LoRA config: r=16, alpha=32, dropout=0.05, targets `q_proj/k_proj/v_proj/o_proj`
- Saves the adapter (a few MB, not the full model) to `./lora-doc-qa`
- On a single 16GB GPU, expect roughly 2.5–3 hours for 3 epochs on 5,000 examples

### Step 4 — Fine-tuned evaluation
```bash
python evaluate.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter_dir ./lora-doc-qa \
  --eval_file ../data/eval.jsonl
```
Outputs a side-by-side table (zero-shot vs fine-tuned) and saves
`eval_results.json`. **The `f1` column is what "91% accuracy" refers to** —
plug in your actual number.

To get the "60% GPU cost cut" / "98% quality retention" comparison, repeat
Step 3–4 once more with `load_in_4bit=False` (full fine-tuning) if you have
the VRAM, and compare training time + F1 against the QLoRA run.

### Step 5 — Retrieval latency benchmark
```bash
cd ../benchmark
python retrieval_benchmark.py --n_chunks 10000 --n_queries 20
```
Compares naive linear-scan cosine search against an ANN-style top-20 +
re-rank approach over a synthetic 10,000-chunk corpus. **This produces the
"55% latency reduction" number.** For a more realistic figure, point this at
your actual indexed documents instead of the synthetic corpus.

### Step 6 — Run the app
```bash
# Terminal 1 (from project root)
uvicorn api.main:app --reload

# Terminal 2
streamlit run app.py
```
Open the Streamlit URL, upload a PDF/DOCX/TXT, and ask questions. Each
response shows the retrieved source chunks for transparency.

The 4-bit base model + LoRA adapter loads once at API startup (singleton in
`api/llm.py`) — subsequent requests reuse it.

---

## 4. Sample Queries (with a sample contract)

- "How many days of written notice are required to terminate the agreement?"
- "Is consent required to assign this agreement to another party?"
- "How long does the confidentiality obligation last after termination?"
- "What is the initial term of the agreement?"

---

## 5. Filling In Your Results Table

After Steps 2–5, copy your real numbers here and use them to update the
resume bullets:

| Metric | Zero-shot | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| Exact Match | | | |
| F1 (≈ accuracy) | | | |
| ROUGE-L | | | |

| Retrieval | Latency (ms/query) |
|---|---|
| Naive linear scan | |
| ANN + re-rank (Chroma) | |
| Reduction | |

---

## 6. Notes

- `data/train.jsonl` and `data/eval.jsonl` currently contain a couple of
  hand-written sample rows so the pipeline can be inspected/tested without
  downloading the full dataset. Run `prepare_dataset.py` to regenerate the
  real 5,000/300 splits.
- Chunking uses a **token-based** splitter (`cl100k_base` via tiktoken) so
  "512-token chunks with 64-token overlap" is literally what's configured —
  not an approximation based on character count.
- The "50+ concurrent users" claim is supported by `/upload` and `/chat`
  running document ingestion and generation in a thread pool
  (`run_in_threadpool`), so the FastAPI event loop isn't blocked — but the
  real ceiling depends on your GPU's throughput. Load-test with a tool like
  `locust` or `hey` against `/chat` to get a real concurrency figure.