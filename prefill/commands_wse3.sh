#!/usr/bin/env bash

set -e

# Uncomment to enable instruction tracing (slows simulation significantly):
# export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace

P=4
DIM=32
SEQ_LEN=32
FFN_DIM=32
N_HEADS=1
N_KV_HEADS=1
HEAD_DIM=32

FABRIC_W=$(($P + 7))
FABRIC_H=$(($P + 2))

dim_p_pe=$(($DIM / $P))
pes_p_head=$(($P / $N_HEADS))
pes_p_kv_head=$(($P / $N_KV_HEADS))
head_dim_p_pe=$(($HEAD_DIM / $P))
seq_len_p_pe=$(($SEQ_LEN / $P))
ffn_dim_p_pe=$(($FFN_DIM / $P))

echo "P: $P, DIM: $DIM, SEQ_LEN: $SEQ_LEN, FFN_DIM: $FFN_DIM"

cslc --arch=wse3 ./src/layout.csl \
    --fabric-dims="$FABRIC_W","$FABRIC_H" --fabric-offsets=4,1 \
    --params=P:"$P",dim_p_pe:"$dim_p_pe",pes_p_head:"$pes_p_head",pes_p_kv_head:"$pes_p_kv_head",head_dim_p_pe:"$head_dim_p_pe",seq_len_p_pe:"$seq_len_p_pe",ffn_dim_p_pe:"$ffn_dim_p_pe" \
    --memcpy --channels=1 -o out

cs_python run.py --name out \
    --P "$P" --dim "$DIM" --seq_len "$SEQ_LEN" --ffn_dim "$FFN_DIM" \
    --head_dim "$HEAD_DIM" --n_heads "$N_HEADS" --n_kv_heads "$N_KV_HEADS" \
    --warmup 0 --repeats 1
