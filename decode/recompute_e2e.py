"""Recompute E2E throughput from existing e2e_verification_term_out.txt
using ACTUAL input tokens (4096/2048) for prefill TPR instead of padded
seq_len. Decode TPR (1/time) is unaffected by the padding bug.

No re-runs needed — only the post-processing formula changes.
"""

import re
from pathlib import Path

DECODE_DIR = Path(__file__).resolve().parent
TERM = DECODE_DIR / "e2e_verification_term_out.txt"
OUT  = DECODE_DIR / "e2e_verification_results.txt"

# Per-config metadata (label -> input_tokens, padded_seq_len, layer_num)
CONFIG = {
    "8B_prefill_4k":  {"input_tokens": 4096, "seq_len": 4620, "layers": 32, "P": 660, "grid": "660x660"},
    "8B_prefill_2k":  {"input_tokens": 2048, "seq_len": 2640, "layers": 32, "P": 660, "grid": "660x660"},
    "8B_decode_4k":   {"seq_len": 4320, "layers": 32, "P": 360, "grid": "360x360"},
    "8B_decode_2k":   {"seq_len": 2160, "layers": 32, "P": 360, "grid": "360x360"},
    "13B_prefill_4k": {"input_tokens": 4096, "seq_len": 4500, "layers": 40, "P": 750, "grid": "750x750"},
    "13B_prefill_2k": {"input_tokens": 2048, "seq_len": 2250, "layers": 40, "P": 750, "grid": "750x750"},
    "13B_decode_4k":  {"seq_len": 4125, "layers": 40, "P": 375, "grid": "375x375"},
    "13B_decode_2k":  {"seq_len": 2250, "layers": 40, "P": 375, "grid": "375x375"},
}

# Paper reference numbers (from WaferLLM OSDI'25)
PAPER = {
    ("8B",  2048,  128): 764.4,
    ("8B",  4096,  128): 604.4,
    ("8B",  2048, 2048): 2370.3,
    ("8B",  4096, 4096): 2459.0,
    ("13B", 2048,  128): 473.9,
    ("13B", 4096,  128): 414.0,
    ("13B", 2048, 2048): 1690.3,
    ("13B", 4096, 4096): 1826.0,
}

PREFILL_RE = re.compile(r"PREFILL\s+(\S+)\s+P=\d+")
DECODE_RE  = re.compile(r"DECODE\s+(\S+)\s+P=\d+")
TIME_RE    = re.compile(r"^Time:\s+([\d.]+)\s+ms")
DEC_TPR_RE = re.compile(r"Decode throughput per request:\s+([\d.]+)")

def parse_term(path: Path) -> dict[str, float]:
    """Return {label -> value}: prefill labels map to time_ms, decode labels to TPR."""
    text = path.read_text()
    out: dict[str, float] = {}
    current = None
    kind = None  # "prefill" | "decode"
    for line in text.splitlines():
        m = PREFILL_RE.search(line)
        if m:
            current = m.group(1); kind = "prefill"; continue
        m = DECODE_RE.search(line)
        if m:
            current = m.group(1); kind = "decode"; continue
        if kind == "prefill" and current:
            t = TIME_RE.match(line)
            if t:
                out[current] = float(t.group(1))
                current = None
        elif kind == "decode" and current:
            t = DEC_TPR_RE.search(line)
            if t:
                out[current] = float(t.group(1))
                current = None
    return out

def prefill_tpr(label: str, time_ms: float) -> float:
    cfg = CONFIG[label]
    return cfg["input_tokens"] / (time_ms * 1e-3 * cfg["layers"])

def e2e(input_tokens: int, new_tokens: int, p_tpr: float, d_tpr: float) -> float:
    return new_tokens / (input_tokens / p_tpr + new_tokens / d_tpr)

def main():
    parsed = parse_term(TERM)

    # Compute corrected prefill TPRs and pull decode TPRs as-is.
    tpr: dict[str, float] = {}
    for label, val in parsed.items():
        if label.endswith("_prefill_4k") or label.endswith("_prefill_2k"):
            tpr[label] = prefill_tpr(label, val)  # val = time_ms
        else:
            tpr[label] = val  # val = decode TPR (already correct)

    P8_PRE_4K  = tpr.get("8B_prefill_4k")
    P8_PRE_2K  = tpr.get("8B_prefill_2k")
    P8_DEC_4K  = tpr.get("8B_decode_4k")
    P8_DEC_2K  = tpr.get("8B_decode_2k")
    P13_PRE_4K = tpr.get("13B_prefill_4k")
    P13_PRE_2K = tpr.get("13B_prefill_2k")
    P13_DEC_4K = tpr.get("13B_decode_4k")
    P13_DEC_2K = tpr.get("13B_decode_2k")

    # E2E scenarios: (model, input_tokens, new_tokens, prefill_tpr, decode_tpr, kernel_label)
    scenarios = [
        ("LLaMA3-8B",   2048,  128,  P8_PRE_2K,  P8_DEC_2K,  "2K prefill + 2K decode"),
        ("LLaMA3-8B",   4096,  128,  P8_PRE_4K,  P8_DEC_4K,  "4K prefill + 4K decode"),
        ("LLaMA3-8B",   2048, 2048,  P8_PRE_2K,  P8_DEC_2K,  "2K prefill + 2K decode"),
        ("LLaMA3-8B",   4096, 4096,  P8_PRE_4K,  P8_DEC_4K,  "4K prefill + 4K decode"),
        ("LLaMA2-13B",  2048,  128,  P13_PRE_2K, P13_DEC_2K, "2K prefill + 2K decode"),
        ("LLaMA2-13B",  4096,  128,  P13_PRE_4K, P13_DEC_4K, "4K prefill + 4K decode"),
        ("LLaMA2-13B",  2048, 2048,  P13_PRE_2K, P13_DEC_2K, "2K prefill + 2K decode"),
        ("LLaMA2-13B",  4096, 4096,  P13_PRE_4K, P13_DEC_4K, "4K prefill + 4K decode"),
    ]

    lines = []
    lines.append("WaferLLM End-to-End Throughput Verification (CORRECTED)")
    lines.append("============================================")
    lines.append("Grid config:  LLaMA3-8B  = 660x660 prefill + 360x360 decode")
    lines.append("              LLaMA2-13B = 750x750 prefill + 375x375 decode")
    lines.append("")
    lines.append("Formula: E2E = new_tokens / (input_tokens/prefill_TPR + new_tokens/decode_TPR)")
    lines.append("Prefill TPR uses ACTUAL input tokens (4096/2048), not padded kernel seq_len.")
    lines.append("Decode TPR formula (1/time) was already correct.")
    lines.append("")
    lines.append("--- Kernel TPR (corrected) ---")
    lines.append("")
    lines.append("+---------------------------+---------+----------+----------+----------+")
    lines.append("| Kernel                    | Grid    | seq_len  | input    | TPR      |")
    lines.append("+---------------------------+---------+----------+----------+----------+")
    rows = [
        ("LLaMA3-8B  prefill 4K-ctx",  "660x660",  4620, 4096, P8_PRE_4K),
        ("LLaMA3-8B  prefill 2K-ctx",  "660x660",  2640, 2048, P8_PRE_2K),
        ("LLaMA3-8B  decode  4K-ctx",  "360x360",  4320, "-",  P8_DEC_4K),
        ("LLaMA3-8B  decode  2K-ctx",  "360x360",  2160, "-",  P8_DEC_2K),
        ("LLaMA2-13B prefill 4K-ctx",  "750x750",  4500, 4096, P13_PRE_4K),
        ("LLaMA2-13B prefill 2K-ctx",  "750x750",  2250, 2048, P13_PRE_2K),
        ("LLaMA2-13B decode  4K-ctx",  "375x375",  4125, "-",  P13_DEC_4K),
        ("LLaMA2-13B decode  2K-ctx",  "375x375",  2250, "-",  P13_DEC_2K),
    ]
    for name, grid, seq_len, in_tok, t in rows:
        in_tok_s = f"{in_tok}" if in_tok != "-" else "-"
        t_s = f"{t:.2f}" if t else "FAILED"
        lines.append(f"| {name:<25} | {grid:<7} | {seq_len:>8} | {in_tok_s:>8} | {t_s:>8} |")
    lines.append("+---------------------------+---------+----------+----------+----------+")
    lines.append("")
    lines.append("--- E2E Throughput (tok/s) — corrected ---")
    lines.append("")
    lines.append("+----------------+-------------+--------+--------+----------+----------+----------------------+")
    lines.append("| Model          | Scenario    |  Ours  |  Paper |  Δ (%)   |  Old     | Kernel used          |")
    lines.append("+----------------+-------------+--------+--------+----------+----------+----------------------+")

    # Old (uncorrected) values from the previous results.txt for comparison
    OLD_E2E = {
        ("LLaMA3-8B",  2048,  128):  747.9,
        ("LLaMA3-8B",  4096,  128):  629.2,
        ("LLaMA3-8B",  2048, 2048): 2036.1,
        ("LLaMA3-8B",  4096, 4096): 2087.0,
        ("LLaMA2-13B", 2048,  128):  499.7,
        ("LLaMA2-13B", 4096,  128):  470.0,
        ("LLaMA2-13B", 2048, 2048): 1534.9,
        ("LLaMA2-13B", 4096, 4096): 1622.2,
    }

    for model, in_tok, out_tok, p, d, kernel in scenarios:
        if p is None or d is None:
            ours_s = "FAILED"
            delta_s = "-"
        else:
            ours = e2e(in_tok, out_tok, p, d)
            ours_s = f"{ours:.1f}"
            mkey = "8B" if "8B" in model else "13B"
            paper = PAPER[(mkey, in_tok, out_tok)]
            delta_s = f"{(ours-paper)/paper*100:+.1f}"
        old = OLD_E2E.get((model, in_tok, out_tok), "-")
        old_s = f"{old:.1f}" if isinstance(old, float) else "-"
        scenario = f"{in_tok}in/{out_tok}"
        mkey = "8B" if "8B" in model else "13B"
        paper_s = f"{PAPER[(mkey, in_tok, out_tok)]:.1f}"
        lines.append(f"| {model:<14} | {scenario:<11} | {ours_s:>6} | {paper_s:>6} | {delta_s:>8} | {old_s:>8} | {kernel:<20} |")
        if model == "LLaMA3-8B" and in_tok == 4096 and out_tok == 4096:
            lines.append("+----------------+-------------+--------+--------+----------+----------+----------------------+")

    lines.append("+----------------+-------------+--------+--------+----------+----------+----------------------+")
    lines.append("")
    lines.append("Δ (%) = (ours − paper) / paper × 100. Negative = below paper, positive = above.")

    out_text = "\n".join(lines)
    OUT.write_text(out_text)
    print(out_text)
    print(f"\nWrote corrected E2E results to {OUT}")

if __name__ == "__main__":
    main()
