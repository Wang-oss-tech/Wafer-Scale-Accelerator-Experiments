"""Benchmark Cerebras Inference (llama3.1-8b) at controlled (input, output) token shapes.

Uses the API's own usage.prompt_tokens to calibrate prompt length, so no HF
tokenizer / gated repo access is needed. Reports TTFT, TPOT, E2E and the
formula  E2E = new / (input/prefill_TPR + new/decode_TPR).
"""

"""
source ~/R_2.9.0/venv_cerebras_pt/bin/activate
pip install --upgrade cerebras_cloud_sdk
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 && export https_proxy=http://proxy.alcf.anl.gov:3128
export CEREBRAS_API_KEY=<your-key>
cd ~/R_2.6.0/csl-examples/experiments/inference_benchmarks

## adjust input and output length 
python ~/R_2.6.0/csl-examples/experiments/inference_benchmarks/bench_cerebras_inference.py \
    --input 4000 --output 4000 \
    --trials 10 --warmup 2 \
    2>&1 | tee ~/R_2.6.0/csl-examples/experiments/inference_benchmarks/results/in4000_out4000.log

"""

import os, time, statistics, argparse, sys
from cerebras.cloud.sdk import Cerebras

MODEL = "llama3.1-8b"


def calibrate_prompt(client, target_input_tokens, max_iter=8):
    """Build a prompt whose API-reported prompt_tokens equals target.

    Strategy: start with filler text proportional to target, send a tiny
    request to read usage.prompt_tokens, then add/remove filler words and
    retry until exact.
    """
    # Start at ~half the target so the initial probe is safely under any
    # context-length limit; bisection-style refinement takes care of the rest.
    # (Old factor //16 overshot to ~3x target and tripped 8K limits at target=4096.)
    base = "Continue counting: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
    msg = base * max(1, target_input_tokens // 32)

    for i in range(max_iter):
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": msg}],
            max_tokens=1,
            temperature=0.0,
        )
        n = r.usage.prompt_tokens
        if n == target_input_tokens:
            return msg, n
        diff = target_input_tokens - n
        if abs(diff) <= 2:
            # close enough — round to within 2 tokens
            return msg, n
        # adjust by approx 1 word per 2 tokens
        word_delta = diff // 2 if abs(diff) > 4 else (1 if diff > 0 else -1)
        if word_delta > 0:
            msg += " " + " ".join(["seventeen"] * word_delta)
        else:
            words = msg.split(" ")
            msg = " ".join(words[: max(1, len(words) + word_delta)])
        print(f"  calibrate iter {i}: have {n}, want {target_input_tokens}, "
              f"adjusting by {word_delta} words", flush=True)

    return msg, n  # best-effort


def run_one(client, prompt, output_tokens):
    """Stream a request; rely on usage.completion_tokens for the real count.
    Returns (ttft, decode_time, total, completion_tokens, finish, prompt_tokens).
    decode_time = total - ttft  (time spent producing tokens after TTFT).
    """
    t0 = time.perf_counter()
    ttft = None
    finish = None
    prompt_tokens = None
    completion_tokens = None
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        # final usage-only chunk has empty choices but populated usage
        if getattr(chunk, "usage", None):
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content and ttft is None:
            ttft = time.perf_counter() - t0
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    total = time.perf_counter() - t0
    decode_time = (total - ttft) if (ttft is not None) else None
    return ttft, decode_time, total, completion_tokens, finish, prompt_tokens


def bench(input_len, output_len, trials=10, warmup=2):
    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
    print(f"\n=== Calibrating prompt for input_len={input_len} ===", flush=True)
    prompt, actual = calibrate_prompt(client, input_len)
    print(f"calibrated: actual prompt_tokens={actual} (target {input_len})", flush=True)

    # Encourage long output: append explicit instruction
    prompt = prompt + "\n\nNow continue counting numbers, one per line, until I tell you to stop."

    print(f"\n=== Warmup {warmup} ===", flush=True)
    for _ in range(warmup):
        run_one(client, prompt, output_len)

    print(f"=== Benchmarking {trials} trials at input={input_len}, output={output_len} ===", flush=True)
    rs = []
    for i in range(trials):
        ttft, dec, total, ntok, finish, ptok = run_one(client, prompt, output_len)
        rs.append((ttft, dec, total, ntok, finish, ptok))
        tpot_ms = (dec / max(ntok - 1, 1)) * 1000 if (dec is not None and ntok) else float("nan")
        print(f"  trial {i+1}: ttft={ttft*1000:.1f}ms  decode={dec*1000:.1f}ms  "
              f"tpot={tpot_ms:.2f}ms  total={total:.3f}s  out_tok={ntok}  "
              f"in_tok={ptok}  finish={finish}", flush=True)

    ttfts   = [r[0] for r in rs if r[0] is not None]
    decodes = [r[1] for r in rs if r[1] is not None]
    totals  = [r[2] for r in rs]
    ns      = [r[3] for r in rs if r[3] is not None]
    pts     = [r[5] for r in rs if r[5] is not None]

    actual_in   = statistics.median(pts)  if pts     else input_len
    med_ttft    = statistics.median(ttfts)
    med_decode  = statistics.median(decodes)
    med_total   = statistics.median(totals)
    med_ntok    = statistics.median(ns)
    avg_ntok    = statistics.mean(ns)

    # TPOT = decode_time / (n_tokens - 1)  (subtract 1 because TTFT covers the first token)
    tpot = med_decode / max(med_ntok - 1, 1)

    prefill_TPR = actual_in / med_ttft
    decode_TPR  = 1 / tpot
    e2e_TPR     = avg_ntok / med_total
    e2e_formula = avg_ntok / (actual_in / prefill_TPR + avg_ntok / decode_TPR)

    print(f"\n=== Results: input={input_len} (actual {actual_in}), output={output_len} ===")
    print(f"  TTFT    median = {med_ttft*1000:8.1f} ms   "
          f"min/max = {min(ttfts)*1000:.1f}/{max(ttfts)*1000:.1f}")
    print(f"  Decode  median = {med_decode*1000:8.1f} ms   "
          f"(time after first token, for {med_ntok:.0f} tokens)")
    print(f"  TPOT           = {tpot*1000:8.2f} ms/tok")
    print(f"  Total   median = {med_total:8.3f}  s")
    print(f"  Out tokens     = {avg_ntok:.1f} avg   (finish_reasons: {set(r[4] for r in rs)})")
    print(f"")
    print(f"  prefill_TPR    = {prefill_TPR:9.1f} tok/s   (input_tokens / TTFT)")
    print(f"  decode_TPR     = {decode_TPR:9.1f} tok/s   (1 / TPOT)")
    print(f"  E2E (measured)            = {e2e_TPR:9.1f} tok/s   (out / total)")
    print(f"  E2E (formula reconstruct) = {e2e_formula:9.1f} tok/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=int, default=2048)
    ap.add_argument("--output", type=int, default=128)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    bench(args.input, args.output, args.trials, args.warmup)
