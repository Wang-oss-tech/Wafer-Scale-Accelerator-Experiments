#!/usr/bin/env python3
"""Find the last non-IDLE executing cycle in a sim.log file."""

import sys
import re

def find_last_active(filename):
    last_active_line = None
    last_active_cycle = 0

    with open(filename, 'r') as f:
        for line in f:
            if '[EX OP]' in line and 'IDLE' not in line:
                match = re.match(r'@(\d+)', line)
                if match:
                    cycle = int(match.group(1))
                    if cycle >= last_active_cycle:
                        last_active_cycle = cycle
                        last_active_line = line.rstrip()

    if last_active_line:
        print(f"Last active cycle: @{last_active_cycle}")
        print(f"Line: {last_active_line}")
    else:
        print("No non-IDLE EX OP found.")

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else 'sim.log'
    find_last_active(filename)
