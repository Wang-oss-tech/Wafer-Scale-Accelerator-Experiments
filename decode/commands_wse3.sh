#!/usr/bin/env bash

set -e

# Uncomment to enable instruction tracing (slows simulation significantly):
# export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace

P=8
BSZ=1
DIM=64
SEQ_LEN=64
FFN_DIM=64
N_HEADS=1
N_KV_HEADS=1
HEAD_DIM=64
GROUP_NUM=2

FABRIC_W=$(($P + 7))
FABRIC_H=$(($P + 2))

dim_p_pe=$(($DIM / $P))
pes_p_head=$(($P / $N_HEADS))
pes_p_kv_head=$(($P / $N_KV_HEADS))
head_dim_p_pe=$(($HEAD_DIM / $P))
seq_len_p_pe=$(($SEQ_LEN / $P))
ffn_dim_p_pe=$(($FFN_DIM / $P))
pe_num_p_group=$(($P / $GROUP_NUM))
root_1st_phase=$((pe_num_p_group / 2))
root_2nd_phase=$((($GROUP_NUM / 2) * pe_num_p_group + root_1st_phase))

echo "P: $P, BSZ: $BSZ, DIM: $DIM, SEQ_LEN: $SEQ_LEN, FFN_DIM: $FFN_DIM"
echo "GROUP_NUM: $GROUP_NUM, pe_num_p_group: $pe_num_p_group, root_1st_phase: $root_1st_phase, root_2nd_phase: $root_2nd_phase"

cslc --arch=wse3 ./src/layout.csl \
    --fabric-dims="$FABRIC_W","$FABRIC_H" --fabric-offsets=4,1 \
    --params=P:"$P",bsz:"$BSZ",dim_p_pe:"$dim_p_pe",pes_p_head:"$pes_p_head",pes_p_kv_head:"$pes_p_kv_head",head_dim_p_pe:"$head_dim_p_pe",seq_len_p_pe:"$seq_len_p_pe",ffn_dim_p_pe:"$ffn_dim_p_pe",pe_num_p_group:"$pe_num_p_group",root_1st_phase:"$root_1st_phase",root_2nd_phase:"$root_2nd_phase" \
    --memcpy --channels=1 -o out

cs_python run.py --name out \
    --P "$P" --bsz "$BSZ" --dim "$DIM" --seq_len "$SEQ_LEN" --ffn_dim "$FFN_DIM" \
    --head_dim "$HEAD_DIM" --n_heads "$N_HEADS" --n_kv_heads "$N_KV_HEADS" \
    --group_num "$GROUP_NUM" \
    --warmup 0 --repeats 1
