#!/usr/bin/env bash

set -e

export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace


cslc --arch=wse3 ./layout.csl --fabric-dims=187,184 --fabric-offsets=4,1 \
--params=P:180,Mt:12,Kt:12,Nt:12 \
--memcpy --channels=1 -o out
cs_python run.py --name out
    