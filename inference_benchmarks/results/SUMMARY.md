# Cerebras Cloud Inference Benchmark — Sweep Summary

Date:        2026-05-06
Model:       llama3.1-8b (Cerebras Cloud Inference, OpenAI-compatible HTTPS API)
Endpoint:    api.cerebras.ai
Methodology: 10 timed trials + 2 warmup per shape, temperature=0.0, stream=True,
             include_usage=True, medians reported, prompt length calibrated against
             API's reported usage.prompt_tokens.

================================================================================
RESULTS TABLE
================================================================================

  Shape (in x out)    TTFT (ms)   TPOT (ms/tok)   Total (s)   prefill_TPR    decode_TPR    E2E
                                                              (tok/s)        (tok/s)       (tok/s)
  ----------------    ---------   -------------   ---------   -----------    ----------    -------
  (2048, 128)             156         0.32          0.20         13,146         3,131         651
  (4096, 128)             162         0.55          0.24         25,431         1,811         546
  (2048, 2048)            162         0.43          1.00         12,651         2,350       1,974
  (4000, 4000)            166         0.41          1.80         24,125         2,427       2,217


  (1) Hard error: context_length_exceeded. Cerebras Cloud's hosted llama3.1-8b is
      configured with an 8,192-token effective context window (input + output
      combined). 4096 + 4096 = 8192 exceeds available room once chat-template
      overhead is added. Use (4000, 4000) as the practical near-substitute.

Actual measured token counts:
  - (2048, 128):  in=2051, out=128
  - (4096, 128):  in=4106, out=128
  - (2048, 2048): in=2051, out=1971   (finish_reason=stop, model emitted EOS early)
  - (4000, 4000): in=4010, out=4000


================================================================================
KEY FINDINGS
================================================================================

1. TPOT IS NOT CONSTANT — IT DEPENDS ON BOTH KV SIZE AND OUTPUT LENGTH
   Two effects compete and the data is not monotonic in KV size alone:

     Shape           Avg KV during decode    TPOT (ms/tok)
     (2048, 128)         ~2,115                  0.32
     (2048, 2048)        ~3,037                  0.43
     (4096, 128)         ~4,170                  0.55
     (4000, 4000)        ~6,010                  0.41   <- larger KV, lower TPOT

   Effect A — Within fixed output length, larger KV slows TPOT.
     Evidence: (2048, 128) -> (4096, 128) at output=128 fixed:
       TPOT 0.32 -> 0.55 (+72%) as input doubles 2048 -> 4096.
     This is the expected KV-bandwidth-bound decode behavior.

   Effect B — Within fixed input length, longer outputs LOWER TPOT.
     Evidence: (4000, 4000) has 6,010-token avg KV but TPOT only 0.41 ms/tok,
     LOWER than (4096, 128)'s 0.55 ms/tok at one-third the average KV size.
     Implication: there is a fixed per-request decode-loop overhead that
     amortizes well over long outputs but penalizes short ones.

2. CSL IMPLICATION
   Cerebras Cloud's hosted decode loop pays a fixed per-request overhead
   that hurts short-output regimes disproportionately. A custom CSL
   implementation that eliminates this overhead has its largest single-point
   opportunity at SHORT-OUTPUT regimes (e.g. interactive chat with brief
   replies), regardless of KV size. Beating the cloud at long-decode
   workloads is harder because that overhead is already amortized away.

3. PREFILL_TPR RISES WITH INPUT LENGTH
   TTFT only grew from 156 -> 162 -> 166 ms across input lengths 2048 -> 4096 -> 4000.
   Network RTT is shape-invariant; wafer prefill is highly parallel. So
   prefill_TPR = input/TTFT appears to grow with input size, but the underlying
   wafer prefill compute time per token is shrinking proportionally less than
   the input grows. TPOT is the cleaner cross-stack metric than prefill_TPR.

4. CLOUD TAIL LATENCY IS REAL
   Three of the runs each had one ~59-second TTFT outlier (queue/rate-limit
   spike). Medians absorb them; mean would be wildly skewed. ALWAYS report
   median + p95, never mean, for cloud APIs.

5. 8K CONTEXT WINDOW IS A HARD WALL
   The hosted llama3.1-8b deployment serves only 8K context, even though the
   model card lists 128K. For (input + output) > 8K, no Cerebras Cloud baseline
   is available; use GPU vLLM or local-cluster modelzoo inference loop instead.


================================================================================
METHODOLOGY (per request)
================================================================================

  GIVEN by the API:
    prompt_tokens     <- chunk.usage.prompt_tokens (final usage-only chunk)
    completion_tokens <- chunk.usage.completion_tokens
    finish_reason     <- chunk.choices[0].finish_reason

  GIVEN by client wall clock (time.perf_counter):
    t0              <- just before opening the stream
    t_first_token   <- when first chunk with non-empty delta.content arrives
    t_end           <- when the stream loop exits

  DERIVED:
    TTFT          = t_first_token - t0
    Total         = t_end - t0
    Decode time   = Total - TTFT
    TPOT          = Decode_time / (completion_tokens - 1)
    prefill_TPR   = prompt_tokens / TTFT
    decode_TPR    = 1 / TPOT
    E2E TPR (measured)  = completion_tokens / Total
    E2E TPR (formula)   = completion_tokens / (prompt_tokens / prefill_TPR
                                                + completion_tokens / decode_TPR)

  Self-consistency: measured E2E vs. formula-reconstructed E2E agreed to <1%
  across all shapes, confirming the per-stage decomposition (prefill + decode)
  has no significant hidden overhead at these workload points.


================================================================================
CAVEATS FOR THE PAPER
================================================================================

  - TTFT includes ALCF -> Cerebras-cloud network RTT (~30-50 ms one-way),
    which inflates prefill_TPR relative to a wafer-only metric. TPOT is the
    cleaner cross-stack apples-to-apples comparison.

  - (2048, 2048) hit finish_reason=stop at avg 1971 tokens, not the requested
    2048. Metrics use actual completion_tokens so the math is consistent, but
    it is not a forced 2048-token decode. To force longer outputs, use prompts
    that resist stopping ("output a 2000-line numbered list, do not stop").

  - (4096, 4096) cannot be measured on this endpoint. For long-context CSL
    comparisons, Cerebras Cloud is not a candidate baseline.

  - All runs are single-tenant and sequential. Multi-tenant behavior under
    concurrent load is unmeasured.


================================================================================
ARTIFACTS
================================================================================

  Per-shape raw logs:
    in2048_out128.log     [run earlier, separate session]
    in4096_out128.log
    in2048_out2048.log
    in4000_out4000.log
    in4096_out4096.log    [error log only - context_length_exceeded]

  This summary:
    SUMMARY.md            (this file)

  Benchmark script:
    ../bench_cerebras_inference.py
