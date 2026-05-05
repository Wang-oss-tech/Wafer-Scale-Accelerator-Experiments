#!/usr/bin/env bash
# Intentionally NOT using `set -e`: we want all 12 configs to run to completion
# even if individual ones fail. Per-config failures are captured in run_one()
# via the if/else around `python compile.py` and the TPR-parse fallback.
set +e

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# Bypass the proxy for internal Cerebras cluster IPs.
# proxy.alcf.anl.gov:3128 has a ~15-min timeout that kills gRPC for large P
# (P=660 decode, P=720 prefill). SDK worker IPs are in 10.0.0.0/8 and 100.64.0.0/10.
export no_proxy="10.,100.64.,localhost,127.0.0.1,.alcf.anl.gov"
export NO_PROXY="$no_proxy"
source ~/R_2.9.0/venv_cerebras_pt/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec > >(tee "$SCRIPT_DIR/results_term_out.txt") 2>&1

RESULTS_FILE="$SCRIPT_DIR/results.txt"

declare -A TPR

run_one() {
    local label="$1"
    local config="$2"

    local P=$(jq -r '.P' "$config")
    local bsz=$(jq -r '.bsz' "$config")
    local dim=$(jq -r '.dim' "$config")
    local n_heads=$(jq -r '.n_heads' "$config")
    local n_kv_heads=$(jq -r '.n_kv_heads' "$config")
    local head_dim=$(jq -r '.head_dim' "$config")
    local seq_len=$(jq -r '.seq_len' "$config")
    local ffn_dim=$(jq -r '.ffn_dim' "$config")
    local layer_num=$(jq -r '.layer_num' "$config")

    echo ""
    echo "=========================================="
    echo "Compiling: $label (P=$P dim=$dim seq_len=$seq_len ffn=$ffn_dim layer_num=$layer_num)"
    echo "=========================================="
    # Prefill compile.py argument order:
    #   P dim seq_len ffn_dim n_heads n_kv_heads head_dim simulator
    if python compile.py "$P" "$dim" "$seq_len" "$ffn_dim" "$n_heads" "$n_kv_heads" "$head_dim" false; then
        echo "Launching: $label"
        # Prefill appliance_launch.py: no --group_num, no --layer_num
        output=$(python appliance_launch.py \
            --P "$P" --dim "$dim" --seq_len "$seq_len" --ffn_dim "$ffn_dim" \
            --head_dim "$head_dim" --n_heads "$n_heads" --n_kv_heads "$n_kv_heads" \
            --warmup 5 --repeats 50 2>&1)
        echo "$output"

        # Prefill run.py prints "Time: X ms" for one layer (averaged over repeats).
        # Full-model TPR = seq_len / (time_ms * 1e-3 * layer_num)
        local time_ms
        time_ms=$(echo "$output" | grep -E "^Time:" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
        if [ -n "$time_ms" ]; then
            tpr=$(python3 -c "print(f'{$seq_len / ($time_ms * 1e-3 * $layer_num):.1f}')")
            TPR["$label"]="$tpr"
            echo ">>> $label time/layer: ${time_ms} ms,  TPR: $tpr tok/s"
        else
            TPR["$label"]="FAILED"
            echo ">>> $label: could not parse Time"
        fi
    else
        TPR["$label"]="FAILED"
        echo ">>> $label: compile failed"
    fi
}

run_one "llama8B_480"    model_config/llama8B_4k_1_480.json
run_one "llama8B_600"    model_config/llama8B_4k_1_600.json
run_one "llama8B_720"    model_config/llama8B_4k_1_720.json
run_one "llama13B_480"   model_config/llama13B_4k_1_480.json
run_one "llama13B_600"   model_config/llama13B_4k_1_600.json
run_one "llama13B_720"   model_config/llama13B_4k_1_720.json
run_one "codellama_480"  model_config/codellama34B_4k_1_480.json
run_one "codellama_600"  model_config/codellama34B_4k_1_600.json
run_one "codellama_720"  model_config/codellama34B_4k_1_720.json
run_one "qwen72B_480"    model_config/qwen72B_4k_1_480.json
run_one "qwen72B_600"    model_config/qwen72B_4k_1_600.json
run_one "qwen72B_720"    model_config/qwen72B_4k_1_720.json

cat > "$RESULTS_FILE" << EOF
Prefill TPR Results (tok/s, full-model)
=======================================
Model            P=480                    P=600                    P=720
Llama3-8B        ${TPR[llama8B_480]:-FAILED}     ${TPR[llama8B_600]:-FAILED}     ${TPR[llama8B_720]:-FAILED}
Llama2-13B       ${TPR[llama13B_480]:-FAILED}    ${TPR[llama13B_600]:-FAILED}    ${TPR[llama13B_720]:-FAILED}
CodeLlama-34B    ${TPR[codellama_480]:-FAILED}   ${TPR[codellama_600]:-FAILED}   ${TPR[codellama_720]:-FAILED}
QWen2-72B        ${TPR[qwen72B_480]:-FAILED}     ${TPR[qwen72B_600]:-FAILED}     ${TPR[qwen72B_720]:-FAILED}
EOF

echo ""
echo "=========================================="
echo "Results written to $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
