#!/bin/bash
# Find potential start_broadcast_async executions

echo "=== Searching for broadcast k=0 around @7138 ==="
awk '/@7130/,/@7150/ {if ($0 ~ /ST16.*2[bf][4-7]|MOVRI.*00c4/) print NR": "$0}' 4_1_instr.log | head -20

echo ""
echo "=== Searching for first configuration sequence ==="
awk '/@7100/,/@7140/ {if ($0 ~ /ST16/ && $0 ~ /2f[4-7]|2b[4-7]/) print NR": "$0}' 4_1_instr.log | head -10
