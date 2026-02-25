#!/usr/bin/env bash

set -e

# export SINGULARITYENV_SIMFABRIC_DEBUG=landing
export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace


cslc --arch=wse2 ./layout.csl --fabric-dims=11,5 \
--fabric-offsets=4,1 --params=kernel_x_dim:4,kernel_y_dim:3,M:6,N:8 \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYH2D_DATA_2_ID:1 \
--params=MEMCPYH2D_DATA_3_ID:6 \
--params=MEMCPYD2H_DATA_1_ID:2 \
-o out --memcpy --channels 1
cs_python run.py --name out
