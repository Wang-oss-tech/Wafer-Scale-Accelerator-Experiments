#!/usr/bin/env bash
set +e

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
source ~/R_2.9.0/venv_cerebras_pt/bin/activate
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy

DECODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFILL_DIR="$(cd "$DECODE_DIR/../prefill" && pwd)"
RESULTS_FILE="$DECODE_DIR/missing_end_to_end_data.txt"

exec > >(tee "$DECODE_DIR/missing_e2e_term_out.txt") 2>&1

declare -A TPR

run_prefill() {
    local label="$1"
    local P="$2" dim="$3" seq_len="$4" ffn_dim="$5" layer_num="$6"

    echo ""
    echo "=========================================="
    echo "Prefill: $label (P=$P dim=$dim seq_len=$seq_len ffn=$ffn_dim layer_num=$layer_num)"
    echo "=========================================="
    cd "$PREFILL_DIR"
    # python compile.py "$P" "$dim" "$seq_len" "$ffn_dim" 1 1 "$dim" false
    if true; then
        output=$(python appliance_launch.py \
            --P "$P" --dim "$dim" --seq_len "$seq_len" --ffn_dim "$ffn_dim" \
            --head_dim "$dim" --n_heads 1 --n_kv_heads 1 \
            --warmup 5 --repeats 50 2>&1)
        echo "$output"
        time_ms=$(echo "$output" | grep -E "^Time:" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
        if [ -n "$time_ms" ]; then
            tpr=$(python3 -c "print(f'{$seq_len / ($time_ms * 1e-3 * $layer_num):.1f}')")
            TPR["$label"]="$tpr"
            echo ">>> $label TPR: $tpr tok/s"
        else
            TPR["$label"]="FAILED"
            echo ">>> $label: could not parse Time"
        fi
    else
        TPR["$label"]="FAILED"
        echo ">>> $label: compile failed"
    fi
}

run_decode() {
    local label="$1"
    local P="$2" dim="$3" seq_len="$4" ffn_dim="$5" group_num="$6" layer_num="$7"

    echo ""
    echo "=========================================="
    echo "Decode: $label (P=$P dim=$dim ffn=$ffn_dim group_num=$group_num layer_num=$layer_num)"
    echo "=========================================="
    cd "$DECODE_DIR"
    # python compile.py "$P" 1 "$dim" 1 1 "$dim" "$seq_len" "$ffn_dim" "$group_num" false
    if true; then
        output=$(python appliance_launch.py \
            --P "$P" --bsz 1 --dim "$dim" --n_heads 1 --n_kv_heads 1 \
            --head_dim "$dim" --seq_len "$seq_len" --ffn_dim "$ffn_dim" \
            --group_num "$group_num" --layer_num "$layer_num" \
            --warmup 5 --repeats 50 2>&1)
        echo "$output"
        tpr=$(echo "$output" | grep "Decode throughput per request" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
        if [ -n "$tpr" ]; then
            TPR["$label"]="$tpr"
            echo ">>> $label TPR: $tpr tok/s"
        else
            TPR["$label"]="FAILED"
            echo ">>> $label: could not parse TPR"
        fi
    else
        TPR["$label"]="FAILED"
        echo ">>> $label: compile failed"
    fi
}

run_prefill "llama8B_prefill_660"  660 5280 5280 14520 32
run_decode  "llama8B_decode_360"   360 4320 4320 14400 20 32
run_prefill "llama13B_prefill_750" 750 6000 6000 14250 40
run_decode  "llama13B_decode_375"  375 5250 4125 13875 15 40

cat > "$RESULTS_FILE" << EOF
Missing End-to-End Data Results
================================
Paper config:
  LLama3-8B:  660x660 prefill + 360x360 decode
  LLama2-13B: 750x750 prefill + 375x375 decode

Formula: E2E = new_tokens / (input_seq / prefill_TPR + new_tokens / decode_TPR)

+----------------------------+--------+-------------------------+
| Config                     | Grid   | TPR (tok/s)             |
+----------------------------+--------+-------------------------+
| LLama3-8B  prefill         | 660x660 | ${TPR[llama8B_prefill_660]:-FAILED}  |
| LLama3-8B  decode          | 360x360 | ${TPR[llama8B_decode_360]:-FAILED}   |
| LLama2-13B prefill         | 750x750 | ${TPR[llama13B_prefill_750]:-FAILED} |
| LLama2-13B decode          | 375x375 | ${TPR[llama13B_decode_375]:-FAILED}  |
+----------------------------+--------+-------------------------+

E2E Verification (LLama3-8B, using 660x660 prefill + 360x360 decode):
  2048in/128out:   $(python3 -c "p=${TPR[llama8B_prefill_660]:-0}; d=${TPR[llama8B_decode_360]:-0}; print(f'{128/(2048/p+128/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 764.4)
  4096in/128out:   $(python3 -c "p=${TPR[llama8B_prefill_660]:-0}; d=${TPR[llama8B_decode_360]:-0}; print(f'{128/(4096/p+128/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 604.4)
  2048in/2048out:  $(python3 -c "p=${TPR[llama8B_prefill_660]:-0}; d=${TPR[llama8B_decode_360]:-0}; print(f'{2048/(2048/p+2048/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 2370.3)
  4096in/4096out:  $(python3 -c "p=${TPR[llama8B_prefill_660]:-0}; d=${TPR[llama8B_decode_360]:-0}; print(f'{4096/(4096/p+4096/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 2459.0)

E2E Verification (LLama2-13B, using 750x750 prefill + 375x375 decode):
  2048in/128out:   $(python3 -c "p=${TPR[llama13B_prefill_750]:-0}; d=${TPR[llama13B_decode_375]:-0}; print(f'{128/(2048/p+128/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 473.9)
  4096in/128out:   $(python3 -c "p=${TPR[llama13B_prefill_750]:-0}; d=${TPR[llama13B_decode_375]:-0}; print(f'{128/(4096/p+128/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 414.0)
  2048in/2048out:  $(python3 -c "p=${TPR[llama13B_prefill_750]:-0}; d=${TPR[llama13B_decode_375]:-0}; print(f'{2048/(2048/p+2048/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 1690.3)
  4096in/4096out:  $(python3 -c "p=${TPR[llama13B_prefill_750]:-0}; d=${TPR[llama13B_decode_375]:-0}; print(f'{4096/(4096/p+4096/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 1826.0)
EOF

echo ""
echo "=========================================="
echo "Results written to $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
