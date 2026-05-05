#!/usr/bin/env bash
set +e

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
source ~/R_2.9.0/venv_cerebras_pt/bin/activate
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy

DECODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFILL_DIR="$(cd "$DECODE_DIR/../prefill" && pwd)"
RESULTS_FILE="$DECODE_DIR/missing_2048_e2e_data.txt"

exec > >(tee "$DECODE_DIR/missing_2048_e2e_term_out.txt") 2>&1

declare -A TPR

run_prefill() {
    local label="$1"
    local P="$2" dim="$3" seq_len="$4" ffn_dim="$5" layer_num="$6"

    echo ""
    echo "=========================================="
    echo "Prefill: $label (P=$P dim=$dim seq_len=$seq_len ffn=$ffn_dim layers=$layer_num)"
    echo "=========================================="
    cd "$PREFILL_DIR"
    python compile.py "$P" "$dim" "$seq_len" "$ffn_dim" 1 1 "$dim" false
    if [ $? -ne 0 ]; then
        TPR["$label"]="COMPILE_FAILED"
        echo ">>> $label: compile failed"
        return
    fi
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
}

run_decode() {
    local label="$1"
    local P="$2" dim="$3" seq_len="$4" ffn_dim="$5" group_num="$6" layer_num="$7"

    echo ""
    echo "=========================================="
    echo "Decode: $label (P=$P dim=$dim seq_len=$seq_len ffn=$ffn_dim group=$group_num layers=$layer_num)"
    echo "=========================================="
    cd "$DECODE_DIR"
    python compile.py "$P" 1 "$dim" 1 1 "$dim" "$seq_len" "$ffn_dim" "$group_num" false
    if [ $? -ne 0 ]; then
        TPR["$label"]="COMPILE_FAILED"
        echo ">>> $label: compile failed"
        return
    fi
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
}

# 2048-input seq_len rounded to nearest P multiple
# LLaMA3-8B:  dim=4620 (P=660), dim=4320 (P=360)  ffn=14520/14400
# LLaMA2-13B: dim=5250 (P=750/375)                  ffn=14250/13875

run_prefill "llama8B_prefill_660_2k"   660 4620 2640 14520 32   # ceil(2048/660)*660=2640
run_decode  "llama8B_decode_360_2k"    360 4320 2160 14400 20 32 # ceil(2048/360)*360=2160
run_prefill "llama13B_prefill_750_2k"  750 5250 2250 14250 40   # ceil(2048/750)*750=2250
run_decode  "llama13B_decode_375_2k"   375 5250 2250 13875 15 40 # ceil(2048/375)*375=2250

cat > "$RESULTS_FILE" << EOF
Missing 2048-Input End-to-End Data
====================================
Paper config:
  LLama3-8B:  660x660 prefill + 360x360 decode
  LLama2-13B: 750x750 prefill + 375x375 decode

seq_len for 2048-input: rounded to nearest multiple of P
  660: ceil(2048/660)*660 = 2640
  750: ceil(2048/750)*750 = 2250
  360: ceil(2048/360)*360 = 2160
  375: ceil(2048/375)*375 = 2250

Formula: E2E = new_tokens / (input_seq / prefill_TPR + new_tokens / decode_TPR)

+----------------------------+---------+-------------------------+
| Config                     | Grid    | TPR (tok/s)             |
+----------------------------+---------+-------------------------+
| LLama3-8B  prefill (2k in) | 660x660 | ${TPR[llama8B_prefill_660_2k]:-FAILED}  |
| LLama3-8B  decode  (2k kv) | 360x360 | ${TPR[llama8B_decode_360_2k]:-FAILED}   |
| LLama2-13B prefill (2k in) | 750x750 | ${TPR[llama13B_prefill_750_2k]:-FAILED} |
| LLama2-13B decode  (2k kv) | 375x375 | ${TPR[llama13B_decode_375_2k]:-FAILED}  |
+----------------------------+---------+-------------------------+

E2E Verification (LLama3-8B):
  2048in/128out:   $(python3 -c "p=${TPR[llama8B_prefill_660_2k]:-0}; d=${TPR[llama8B_decode_360_2k]:-0}; print(f'{128/(2048/p+128/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 764.4)
  2048in/2048out:  $(python3 -c "p=${TPR[llama8B_prefill_660_2k]:-0}; d=${TPR[llama8B_decode_360_2k]:-0}; print(f'{2048/(2048/p+2048/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 2370.3)

E2E Verification (LLama2-13B):
  2048in/128out:   $(python3 -c "p=${TPR[llama13B_prefill_750_2k]:-0}; d=${TPR[llama13B_decode_375_2k]:-0}; print(f'{128/(2048/p+128/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 473.9)
  2048in/2048out:  $(python3 -c "p=${TPR[llama13B_prefill_750_2k]:-0}; d=${TPR[llama13B_decode_375_2k]:-0}; print(f'{2048/(2048/p+2048/d):.1f}' if p and d else 'FAILED')" 2>/dev/null) tok/s  (paper: 1690.3)
EOF

echo ""
echo "=========================================="
echo "Results written to $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
