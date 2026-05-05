#!/usr/bin/env bash
# End-to-end throughput verification for LLaMA3-8B and LLaMA2-13B.
# Runs all 8 kernel configs (2 models x 2 phases x 2 context lengths)
# and computes the 8 E2E scenarios from the WaferLLM paper.
#
# Model dimensions (rounded to nearest multiple of P):
#   LLaMA3-8B : dim=4096, ffn=14336  -> at P=660: dim=4620, ffn=14520
#                                     -> at P=360: dim=4320, ffn=14400
#   LLaMA2-13B: dim=5120, ffn=13824  -> at P=750: dim=5250, ffn=14250
#                                     -> at P=375: dim=5250, ffn=13875
#
# seq_len rounded to nearest multiple of P:
#   4096-context: 660->4620, 750->4500, 360->4320, 375->4125
#   2048-context: 660->2640, 750->2250, 360->2160, 375->2250

set +e

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
source ~/R_2.9.0/venv_cerebras_pt/bin/activate
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy

DECODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFILL_DIR="$(cd "$DECODE_DIR/../prefill" && pwd)"
RESULTS_FILE="$DECODE_DIR/e2e_verification_results.txt"

exec > >(tee "$DECODE_DIR/e2e_verification_term_out.txt") 2>&1

declare -A TPR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_prefill() {
    local label="$1" P="$2" dim="$3" seq_len="$4" ffn_dim="$5" layer_num="$6"
    echo ""
    echo "=========================================="
    echo "PREFILL $label  P=$P  dim=$dim  seq_len=$seq_len  ffn=$ffn_dim  layers=$layer_num"
    echo "=========================================="
    cd "$PREFILL_DIR"
    python compile.py "$P" "$dim" "$seq_len" "$ffn_dim" 1 1 "$dim" false
    if [ $? -ne 0 ]; then
        TPR["$label"]="COMPILE_FAILED"; echo ">>> COMPILE FAILED"; return
    fi
    local output
    output=$(python appliance_launch.py \
        --P "$P" --dim "$dim" --seq_len "$seq_len" --ffn_dim "$ffn_dim" \
        --head_dim "$dim" --n_heads 1 --n_kv_heads 1 \
        --warmup 5 --repeats 50 2>&1)
    echo "$output"
    local time_ms
    time_ms=$(echo "$output" | grep -E "^Time:" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
    if [ -n "$time_ms" ]; then
        local tpr
        tpr=$(python3 -c "print(f'{$seq_len / ($time_ms * 1e-3 * $layer_num):.2f}')")
        TPR["$label"]="$tpr"
        echo ">>> $label TPR: $tpr tok/s"
    else
        TPR["$label"]="FAILED"; echo ">>> Could not parse Time"
    fi
}

run_decode() {
    local label="$1" P="$2" dim="$3" seq_len="$4" ffn_dim="$5" group_num="$6" layer_num="$7"
    echo ""
    echo "=========================================="
    echo "DECODE  $label  P=$P  dim=$dim  seq_len=$seq_len  ffn=$ffn_dim  group=$group_num  layers=$layer_num"
    echo "=========================================="
    cd "$DECODE_DIR"
    python compile.py "$P" 1 "$dim" 1 1 "$dim" "$seq_len" "$ffn_dim" "$group_num" false
    if [ $? -ne 0 ]; then
        TPR["$label"]="COMPILE_FAILED"; echo ">>> COMPILE FAILED"; return
    fi
    local output
    output=$(python appliance_launch.py \
        --P "$P" --bsz 1 --dim "$dim" --n_heads 1 --n_kv_heads 1 \
        --head_dim "$dim" --seq_len "$seq_len" --ffn_dim "$ffn_dim" \
        --group_num "$group_num" --layer_num "$layer_num" \
        --warmup 5 --repeats 50 2>&1)
    echo "$output"
    local tpr
    tpr=$(echo "$output" | grep "Decode throughput per request" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
    if [ -n "$tpr" ]; then
        TPR["$label"]="$tpr"
        echo ">>> $label TPR: $tpr tok/s"
    else
        TPR["$label"]="FAILED"; echo ">>> Could not parse Decode throughput per request"
    fi
}

e2e() {
    # e2e <in> <out> <prefill_tpr> <decode_tpr>
    python3 -c "
p=${3:-0}; d=${4:-0}
if p and d:
    print(f'{$2/($1/p+$2/d):.1f}')
else:
    print('FAILED')
" 2>/dev/null
}

# ---------------------------------------------------------------------------
# LLaMA3-8B  (P=660 prefill, P=360 decode)
# ---------------------------------------------------------------------------

run_prefill "8B_prefill_4k"  660 4620 4620 14520 32   # 4096-input: ceil(4096/660)*660=4620
run_prefill "8B_prefill_2k"  660 4620 2640 14520 32   # 2048-input: ceil(2048/660)*660=2640
run_decode  "8B_decode_4k"   360 4320 4320 14400 20 32 # 4096 KV:   ceil(4096/360)*360=4320
run_decode  "8B_decode_2k"   360 4320 2160 14400 20 32 # 2048 KV:   ceil(2048/360)*360=2160

# ---------------------------------------------------------------------------
# LLaMA2-13B  (P=750 prefill, P=375 decode)
# ---------------------------------------------------------------------------

run_prefill "13B_prefill_4k" 750 5250 4500 14250 40   # 4096-input: ceil(4096/750)*750=4500
run_prefill "13B_prefill_2k" 750 5250 2250 14250 40   # 2048-input: ceil(2048/750)*750=2250
run_decode  "13B_decode_4k"  375 5250 4125 13875 15 40 # 4096 KV:   ceil(4096/375)*375=4125
run_decode  "13B_decode_2k"  375 5250 2250 13875 15 40 # 2048 KV:   ceil(2048/375)*375=2250

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

P8_PRE_4K=${TPR[8B_prefill_4k]:-FAILED}
P8_PRE_2K=${TPR[8B_prefill_2k]:-FAILED}
P8_DEC_4K=${TPR[8B_decode_4k]:-FAILED}
P8_DEC_2K=${TPR[8B_decode_2k]:-FAILED}
P13_PRE_4K=${TPR[13B_prefill_4k]:-FAILED}
P13_PRE_2K=${TPR[13B_prefill_2k]:-FAILED}
P13_DEC_4K=${TPR[13B_decode_4k]:-FAILED}
P13_DEC_2K=${TPR[13B_decode_2k]:-FAILED}

cat > "$RESULTS_FILE" << EOF
WaferLLM End-to-End Throughput Verification
============================================
Grid config:  LLaMA3-8B  = 660x660 prefill + 360x360 decode
              LLaMA2-13B = 750x750 prefill + 375x375 decode

Formula: E2E = new_tokens / (input_tokens/prefill_TPR + new_tokens/decode_TPR)
         Uses context-matched prefill and decode TPR (2K kernel for 2K input, etc.)

--- Kernel TPR ---

+---------------------------+---------+---------+------------------+
| Kernel                    | Grid    | seq_len | TPR (tok/s)      |
+---------------------------+---------+---------+------------------+
| LLaMA3-8B  prefill 4K-ctx | 660x660 |    4620 | $P8_PRE_4K
| LLaMA3-8B  prefill 2K-ctx | 660x660 |    2640 | $P8_PRE_2K
| LLaMA3-8B  decode  4K-ctx | 360x360 |    4320 | $P8_DEC_4K
| LLaMA3-8B  decode  2K-ctx | 360x360 |    2160 | $P8_DEC_2K
+---------------------------+---------+---------+------------------+
| LLaMA2-13B prefill 4K-ctx | 750x750 |    4500 | $P13_PRE_4K
| LLaMA2-13B prefill 2K-ctx | 750x750 |    2250 | $P13_PRE_2K
| LLaMA2-13B decode  4K-ctx | 375x375 |    4125 | $P13_DEC_4K
| LLaMA2-13B decode  2K-ctx | 375x375 |    2250 | $P13_DEC_2K
+---------------------------+---------+---------+------------------+

--- E2E Throughput (tok/s) ---

+----------------+-------------+----------------+------------------+------------------+
| Model          | Scenario    |   Ours (tok/s) |   Paper (tok/s)  | Kernel used      |
+----------------+-------------+----------------+------------------+------------------+
| LLaMA3-8B      | 2048in/128  | $(e2e 2048 128 $P8_PRE_2K $P8_DEC_2K) | 764.4  | 2K prefill + 2K decode |
| LLaMA3-8B      | 4096in/128  | $(e2e 4096 128 $P8_PRE_4K $P8_DEC_4K) | 604.4  | 4K prefill + 4K decode |
| LLaMA3-8B      | 2048in/2048 | $(e2e 2048 2048 $P8_PRE_2K $P8_DEC_2K) | 2370.3 | 2K prefill + 2K decode |
| LLaMA3-8B      | 4096in/4096 | $(e2e 4096 4096 $P8_PRE_4K $P8_DEC_4K) | 2459.0 | 4K prefill + 4K decode |
+----------------+-------------+----------------+------------------+------------------+
| LLaMA2-13B     | 2048in/128  | $(e2e 2048 128 $P13_PRE_2K $P13_DEC_2K) | 473.9  | 2K prefill + 2K decode |
| LLaMA2-13B     | 4096in/128  | $(e2e 4096 128 $P13_PRE_4K $P13_DEC_4K) | 414.0  | 4K prefill + 4K decode |
| LLaMA2-13B     | 2048in/2048 | $(e2e 2048 2048 $P13_PRE_2K $P13_DEC_2K) | 1690.3 | 2K prefill + 2K decode |
| LLaMA2-13B     | 4096in/4096 | $(e2e 4096 4096 $P13_PRE_4K $P13_DEC_4K) | 1826.0 | 4K prefill + 4K decode |
+----------------+-------------+----------------+------------------+------------------+
EOF

echo ""
echo "=========================================="
echo "Results written to $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
