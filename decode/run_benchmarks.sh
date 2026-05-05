#!/usr/bin/env bash
# Intentionally NOT using `set -e`: we want all 12 configs to run to completion
# even if individual ones fail. Per-config failures are captured in run_one()
# via the if/else around `python compile.py` and the TPR-parse fallback, which
# both write TPR["$label"]="FAILED" so the final results table stays well-formed.
set +e

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
source ~/R_2.9.0/venv_cerebras_pt/bin/activate

# Unset proxy after venv activation so gRPC (compile + launch) connects
# directly to the cluster instead of routing through the ALCF proxy,
# which times out ~15 min and kills P=660 compiles (~20 min).
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy

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
    local group_num=$(jq -r '.group_num' "$config")
    local layer_num=$(jq -r '.layer_num' "$config")

    echo ""
    echo "=========================================="
    echo "Compiling: $label (P=$P dim=$dim ffn=$ffn_dim layer_num=$layer_num)"
    echo "=========================================="
    if python compile.py "$P" "$bsz" "$dim" "$n_heads" "$n_kv_heads" "$head_dim" "$seq_len" "$ffn_dim" "$group_num" false; then
        echo "Launching: $label"
        output=$(python appliance_launch.py \
            --P "$P" --bsz "$bsz" --dim "$dim" --seq_len "$seq_len" --ffn_dim "$ffn_dim" \
            --head_dim "$head_dim" --n_heads "$n_heads" --n_kv_heads "$n_kv_heads" \
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

run_one "llama8B_420"   model_config/llama8B_4k_1_420.json
run_one "llama8B_540"   model_config/llama8B_4k_1_540.json
run_one "llama8B_660"   model_config/llama8B_4k_1_660.json
run_one "llama13B_420"  model_config/llama13B_4k_1_420.json
run_one "llama13B_540"  model_config/llama13B_4k_1_540.json
run_one "llama13B_660"  model_config/llama13B_4k_1_660.json
run_one "codellama_420" model_config/codellama34B_4k_1_420.json
run_one "codellama_540" model_config/codellama34B_4k_1_540.json
run_one "codellama_660" model_config/codellama34B_4k_1_660.json
run_one "qwen72B_420"   model_config/qwen72B_4k_1_420.json
run_one "qwen72B_540"   model_config/qwen72B_4k_1_540.json
run_one "qwen72B_660"   model_config/qwen72B_4k_1_660.json

cat > "$RESULTS_FILE" << EOF
Decode TPR Results (tok/s)
==========================
Model            P=420                    P=540                    P=660
Llama3-8B        ${TPR[llama8B_420]:-FAILED}     ${TPR[llama8B_540]:-FAILED}     ${TPR[llama8B_660]:-FAILED}
Llama2-13B       ${TPR[llama13B_420]:-FAILED}    ${TPR[llama13B_540]:-FAILED}    ${TPR[llama13B_660]:-FAILED}
CodeLlama-34B    ${TPR[codellama_420]:-FAILED}   ${TPR[codellama_540]:-FAILED}   ${TPR[codellama_660]:-FAILED}
QWen2-72B        ${TPR[qwen72B_420]:-FAILED}     ${TPR[qwen72B_540]:-FAILED}     ${TPR[qwen72B_660]:-FAILED}
EOF

echo ""
echo "=========================================="
echo "Results written to $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
