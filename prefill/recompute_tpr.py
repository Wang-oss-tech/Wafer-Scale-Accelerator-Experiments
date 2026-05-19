"""Recompute prefill TPR using ACTUAL input tokens (4096) instead of the
padded seq_len. Reads time_ms values from results_term_out.txt and rewrites
results.txt with the corrected numbers.

No kernel re-runs needed — only changes the post-processing formula:
    OLD: tpr = padded_seq_len / (time_ms * 1e-3 * layer_num)
    NEW: tpr = actual_input_tokens / (time_ms * 1e-3 * layer_num)
"""

import re
import sys
from pathlib import Path

ACTUAL_INPUT_TOKENS = 4096   # all current prefill configs target 4096-token input

SCRIPT_DIR = Path(__file__).resolve().parent
TERM_OUT = SCRIPT_DIR / "results_term_out.txt"
RESULTS  = SCRIPT_DIR / "results.txt"

# Map config label -> (model_name, P) for the result-table grouping.
LABEL_RE = re.compile(
    r"Compiling:\s+(?P<label>\S+)\s+\(P=(?P<P>\d+)\s+dim=(?P<dim>\d+)\s+"
    r"seq_len=(?P<seq_len>\d+)\s+ffn=(?P<ffn>\d+)\s+layer_num=(?P<layers>\d+)\)"
)
TIME_RE = re.compile(r"^Time:\s+([\d.]+)\s+ms")

# Pretty model names for the table.
MODEL_NAMES = {
    "llama8B":   "Llama3-8B",
    "llama13B":  "Llama2-13B",
    "codellama": "CodeLlama-34B",
    "qwen72B":   "QWen2-72B",
}

def parse_term_output(path: Path):
    """Yield (label, P, padded_seq_len, layer_num, time_ms) tuples in order."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    text = path.read_text()
    # Walk through the file, tracking the most recent "Compiling: ..." block.
    # The next "Time: X ms" line that appears after a Compiling header belongs
    # to that config (the launch sequence prints config -> Time -> next config).
    current = None
    for line in text.splitlines():
        m = LABEL_RE.search(line)
        if m:
            current = m.groupdict()
            current["P"] = int(current["P"])
            current["seq_len"] = int(current["seq_len"])
            current["layers"] = int(current["layers"])
            continue
        t = TIME_RE.match(line)
        if t and current is not None:
            yield (
                current["label"],
                current["P"],
                current["seq_len"],
                current["layers"],
                float(t.group(1)),
            )
            current = None  # consume this Time for this config

def model_from_label(label: str) -> str:
    # labels look like "llama8B_480" or "codellama_600"
    base = label.rsplit("_", 1)[0]
    return MODEL_NAMES.get(base, base)

def main():
    rows = list(parse_term_output(TERM_OUT))
    if not rows:
        sys.exit("ERROR: no (config, Time) pairs parsed from term output")

    # Group: model_name -> {P -> tpr}
    table: dict[str, dict[int, float]] = {}
    detail: list[str] = []
    for label, P, padded_seq_len, layers, time_ms in rows:
        model = model_from_label(label)
        old_tpr = padded_seq_len / (time_ms * 1e-3 * layers)
        new_tpr = ACTUAL_INPUT_TOKENS / (time_ms * 1e-3 * layers)
        table.setdefault(model, {})[P] = new_tpr
        detail.append(
            f"  {label:<22} P={P:<3} padded_seq_len={padded_seq_len}  "
            f"time={time_ms:.4f} ms  layers={layers}  "
            f"old_TPR={old_tpr:8.1f}  new_TPR={new_tpr:8.1f}  "
            f"(scaled by {ACTUAL_INPUT_TOKENS}/{padded_seq_len}={ACTUAL_INPUT_TOKENS/padded_seq_len:.4f})"
        )

    # Write a results.txt with the corrected numbers in the same table layout
    # as the previous version, and append a per-config detail section.
    Ps = sorted({P for d in table.values() for P in d.keys()})
    lines = []
    lines.append(f"Prefill TPR Results (tok/s, full-model, input_tokens={ACTUAL_INPUT_TOKENS})")
    lines.append("=" * 70)
    lines.append(f"Per-config TPR uses actual input tokens ({ACTUAL_INPUT_TOKENS}), not padded seq_len.")
    lines.append("")
    header = f"{'Model':<16}" + "".join(f"P={P:<22}" for P in Ps)
    lines.append(header)
    for model in ["Llama3-8B", "Llama2-13B", "CodeLlama-34B", "QWen2-72B"]:
        if model not in table:
            continue
        row = f"{model:<16}"
        for P in Ps:
            v = table[model].get(P)
            row += f"{v:<24.1f}" if v is not None else f"{'-':<24}"
        lines.append(row)
    lines.append("")
    lines.append("Per-config detail:")
    lines.append("-" * 70)
    lines.extend(detail)
    lines.append("")

    out = "\n".join(lines)
    RESULTS.write_text(out)
    print(out)
    print(f"\nWrote corrected results to {RESULTS}")

if __name__ == "__main__":
    main()
