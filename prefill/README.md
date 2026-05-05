# WaferLLM Prefill — WSE-3 Port

WSE-3 port of the WaferLLM Prefill kernel from `WaferLLM/Prefill/WSE-2/`.

Implements one full transformer layer (attention + FFN) for prefill inference on a P×P PE grid.

## Parameters

| Name | Description |
|---|---|
| P | PE grid size (P×P total PEs) |
| dim | Model hidden dimension |
| seq_len | Sequence length |
| ffn_dim | FFN intermediate dimension |
| head_dim | Attention head dimension |
| n_heads | Number of attention heads |
| n_kv_heads | Number of KV heads (GQA support) |

Derived: `dim_p_pe = dim/P`, `seq_len_p_pe = seq_len/P`, `ffn_dim_p_pe = ffn_dim/P`

## Compile and run (simulator)

```bash
./commands_wse3.sh
```

Default test: P=4, dim=32, seq_len=32, ffn_dim=32, head_dim=32, n_heads=1.

Manual:
```bash
cslc --arch=wse3 ./src/layout.csl \
  --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,dim_p_pe:8,pes_p_head:4,pes_p_kv_head:4,head_dim_p_pe:8,seq_len_p_pe:8,ffn_dim_p_pe:8 \
  --memcpy --channels=1 -o out

cs_python run.py --name out --P 4 --dim 32 --seq_len 32 --ffn_dim 32 \
  --head_dim 32 --n_heads 1 --n_kv_heads 1 --warmup 0 --repeats 1
```

## Compile for hardware (appliance)

```bash
cs_python compile.py 4 32 32 32 1 1 32 false
```

Arguments: P dim seq_len ffn_dim n_heads n_kv_heads head_dim simulator(true/false)

## WSE-3 migration TODOs

1. **`simd_mode` in `src/comm_lib/comm_pe.csl`**: All `simd_mode = .{ .simd_32 = true }` and `simd_mode = .{ .simd_64 = true }` fields are not supported in the local simulator. For hardware compilation, `compile.py` automatically patches these to `.simd_max = true`. For simulation, consider removing them (they are throughput hints, not required for correctness).

2. **No WSE-3 version exists upstream**: This directory is the primary WSE-3 implementation. The WSE-2 source is at `../gemm/WaferLLM/Prefill/WSE-2/`.

3. **`--arch=wse3`**: All compile commands use `--arch=wse3`. The fabric dim formula `P+7, P+2` is shared between WSE-2 and WSE-3 for simulation.
