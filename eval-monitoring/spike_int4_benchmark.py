"""
INT4 Marlin latency spike — Phase 0 of experiment/llm-int4-marlin.

PURPOSE
-------
Picks the fastest in-process INT4 loader that (a) engages Marlin kernels and
(b) does NOT downgrade the shared `transformers` install (which also serves
Whisper ASR and the Llama 3.1 chat template).

HOW TO RUN
----------
1. Start a fresh Colab Pro runtime (L4 or A100).
2. Install prerequisites in Cell 4 / a separate install cell — ADD these lines:
       !pip install -q compressed-tensors
       # !pip install -q gptqmodel   # uncomment to also test Arm 2
   Do NOT install autoawq — it downgrades transformers.
3. Run this script BEFORE Cell 5 (before import app / model loading).
   In a notebook cell:
       import subprocess, sys
       subprocess.run([sys.executable, "eval-monitoring/spike_int4_benchmark.py"])
   Or copy the code directly into a cell.
4. Note the tok/s for each arm and the transformers version reported at the end.
5. Pick the winner (fastest arm that did not change the transformers version).
6. Report back — Phase 1 applies the winner to app.py + requirements.txt.

CANDIDATES
----------
  Arm 0: bnb-nf4 (current production config)           -- baseline
  Arm 1: compressed-tensors W4A16 (RedHatAI/NeuralMagic)  -- preferred (maintained, no downgrade)
  Arm 2: gptqmodel GPTQ INT4 (TechxGenus)               -- optional, uncomment below

DECISION RULE
-------------
  Accept a candidate if:
    - tok/s is clearly higher than Arm 0 (Marlin is engaged)
    - transformers version is unchanged after the install
  If both Arm 1 and Arm 2 pass, pick the faster one.
  If neither pass, fall back to classic AWQ + autoawq (accept the transformers
  downgrade, reinstall transformers afterwards) -- see comment at bottom.
"""

import os, sys, time, gc
import torch
import transformers

HF_TOKEN = os.environ.get("HF_TOKEN")  # set via Colab secrets in normal notebook setup
if not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass

BENCH_PROMPT = (
    "तू हमके बताव कि धान के फसल में पीला पत्ता किस कारण होला आउर ओकर का उपाय बा?"
)
N_TOKENS = 150  # fixed token budget; greedy; no sampling

_tf_version_before = transformers.__version__
print(f"transformers version at start: {_tf_version_before}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  ({props.total_memory/1e9:.1f} GB)  compute={props.major}.{props.minor}")


def bench(model_id, load_kwargs, label, tok_kwargs=None):
    """Load model, warmup, benchmark, free. Returns (tok/s, vram_gb)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\n{'='*62}")
    print(f"Testing : {label}")
    print(f"Checkpoint: {model_id}")

    kw = {"token": HF_TOKEN} if HF_TOKEN else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kw, **(tok_kwargs or {}))

    t_load = time.time()
    m = AutoModelForCausalLM.from_pretrained(model_id, **kw, **load_kwargs)
    m.eval()
    load_s = time.time() - t_load
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Load time : {load_s:.1f}s")
    print(f"  VRAM used : {vram_gb:.1f} GB")

    # detect quant method from config
    qc = getattr(m.config, "quantization_config", None)
    if qc:
        qm = (getattr(qc, "quant_type", None) or
              getattr(qc, "quant_method", None) or
              type(qc).__name__)
        print(f"  Quant method : {qm}")

    dev = next(m.parameters()).device
    pad_id = tok.eos_token_id
    inputs = tok(BENCH_PROMPT, return_tensors="pt").to(dev)

    # warmup — CUDA graph capture / Marlin autotune happens on first generate
    with torch.no_grad():
        m.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=pad_id)

    # timed benchmark run
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=N_TOKENS, do_sample=False, pad_token_id=pad_id)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    ntok = out.shape[1] - inputs["input_ids"].shape[1]
    tps = ntok / elapsed
    print(f"  Generated : {ntok} tokens in {elapsed:.2f}s")
    print(f"  Throughput: >>> {tps:.1f} tok/s <<<   (baseline will be shown in summary)")

    del m, tok, inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    return tps, vram_gb


results = {}

# -----------------------------------------------------------------------
# Arm 0: bnb-nf4 — current production config (baseline)
# -----------------------------------------------------------------------
from transformers import BitsAndBytesConfig
bnb_tps, bnb_vram = bench(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_kwargs={
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        "device_map": "auto",
    },
    label="[ARM 0] bnb nf4 — BASELINE",
)
results["bnb_nf4"] = (bnb_tps, bnb_vram)

# -----------------------------------------------------------------------
# Arm 1: compressed-tensors W4A16  (RedHatAI / NeuralMagic)
# Needs:  !pip install -q compressed-tensors
# Ungated: uses standard Llama 3.1 community license.
# -----------------------------------------------------------------------
ct_tps, ct_vram = bench(
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
    load_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
    label="[ARM 1] compressed-tensors W4A16 — RedHatAI",
)
results["compressed_tensors_w4a16"] = (ct_tps, ct_vram)

# -----------------------------------------------------------------------
# Arm 2: gptqmodel GPTQ INT4 — optional, uncomment to test
# Needs:  !pip install -q gptqmodel
# Checkpoint: verify this repo exists on HF before running.
# -----------------------------------------------------------------------
# gptq_tps, gptq_vram = bench(
#     "TechxGenus/Meta-Llama-3.1-8B-Instruct-GPTQ",
#     load_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
#     label="[ARM 2] gptqmodel GPTQ INT4 — TechxGenus",
# )
# results["gptq_gptqmodel"] = (gptq_tps, gptq_vram)

# -----------------------------------------------------------------------
# Arm 3: classic AWQ — fallback only (autoawq downgrades transformers)
# If choosing this arm, plan to reinstall transformers afterwards:
#   !pip install -q "transformers>=4.49,<4.53"
# -----------------------------------------------------------------------
# awq_tps, awq_vram = bench(
#     "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
#     load_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
#     label="[ARM 3] AWQ — hugging-quants (FALLBACK, may downgrade transformers)",
# )
# results["awq_hugging_quants"] = (awq_tps, awq_vram)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*62}")
print("SUMMARY")
print(f"{'Arm':<40} {'tok/s':>8}  {'vs baseline':>12}  {'VRAM':>8}")
print("-" * 62)
baseline_tps = results["bnb_nf4"][0]
for name, (tps, vram) in results.items():
    speedup = f"{tps/baseline_tps:.2f}x" if name != "bnb_nf4" else "(baseline)"
    print(f"  {name:<38} {tps:>8.1f}  {speedup:>12}  {vram:>6.1f} GB")

print()
import importlib; importlib.reload(transformers)
tf_version_after = transformers.__version__
print(f"transformers version: {_tf_version_before}  -->  {tf_version_after}")
if _tf_version_before != tf_version_after:
    print("  *** WARNING: transformers was downgraded! This loader is not safe for this backend. ***")
else:
    print("  OK: transformers version unchanged.")

print()
print("NEXT STEP")
print("  Report the tok/s numbers + transformers version check back,")
print("  and Phase 1 will apply the winner to app.py + requirements.txt.")
