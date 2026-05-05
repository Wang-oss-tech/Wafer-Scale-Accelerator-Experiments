#!/usr/bin/env bash
# 2.5D mini-MeshGEMM Phase 1 build + run (per-layer Cannon, host-side reduce).
# Usage: ./commands_wse3.sh <P_phys> <M> <K> <N> <sqrt_c>
# Example (smallest sane case, sqrt_c=2 -> c=4 layers, q=2 PEs/side per layer):
#   ./commands_wse3.sh 4 4 16 4 2

set -e

P_phys=${1:-4}
M=${2:-4}
K=${3:-16}
N=${4:-4}
sqrt_c=${5:-2}

c=$((sqrt_c * sqrt_c))
q=$((P_phys / sqrt_c))

if [ $((sqrt_c * q)) -ne ${P_phys} ]; then
    echo "ERROR: P_phys ($P_phys) must equal sqrt_c ($sqrt_c) * q ($q)" >&2
    exit 1
fi
if [ $((M % q)) -ne 0 ]; then echo "ERROR: M ($M) not divisible by q ($q)" >&2; exit 1; fi
if [ $((N % q)) -ne 0 ]; then echo "ERROR: N ($N) not divisible by q ($q)" >&2; exit 1; fi
if [ $((K % (c * q))) -ne 0 ]; then echo "ERROR: K ($K) not divisible by c*q ($((c*q)))" >&2; exit 1; fi

Mt=$((M / q))
Kt_layer=$((K / (c * q)))
Nt=$((N / q))

fabric_w=$((P_phys + 7))
fabric_h=$((P_phys + 2))

echo "P_phys=$P_phys, sqrt_c=$sqrt_c, c=$c, q=$q"
echo "M=$M, K=$K, N=$N  ->  Mt=$Mt, Kt_layer=$Kt_layer, Nt=$Nt"
echo "fabric=${fabric_w}x${fabric_h}"

cslc --arch=wse3 ./src/layout.csl \
    --fabric-dims=${fabric_w},${fabric_h} --fabric-offsets=4,1 \
    --params=P_phys:${P_phys},sqrt_c:${sqrt_c},q:${q},Mt:${Mt},Kt_layer:${Kt_layer},Nt:${Nt} \
    -o out --memcpy --channels 1

cs_python ./launch_sim.py --P_phys ${P_phys} --M ${M} --K ${K} --N ${N} --sqrt_c ${sqrt_c}
