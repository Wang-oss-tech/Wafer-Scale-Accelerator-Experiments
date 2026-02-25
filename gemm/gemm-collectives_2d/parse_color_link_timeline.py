#!/usr/bin/env python3
"""
Parse log file to track changes in (color, link) pairs and output their time ranges.
Reports whenever the (color, link) pair changes.
"""

import re
import sys

def parse_log_file(log_file_path):
    """
    Parse the log file and extract timeline information for (color, link) pairs.
    
    Returns:
        - pair_ranges: List of (start_time, end_time, color, link) tuples
    """
    pair_ranges = []
    
    # Pattern to match log entries
    # Format: @timestamp P4.1 (hwtile) landing C<color> from link <direction>, ...
    pattern = re.compile(r'@(\d+)\s+.*?landing\s+(C\d+)\s+from\s+link\s+([A-Z])')
    
    current_pair = None  # (color, link)
    current_start = None
    current_end = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                timestamp = int(match.group(1))
                color = match.group(2)
                link_direction = match.group(3)
                
                new_pair = (color, link_direction)
                
                if current_pair is None:
                    # First pair encountered
                    current_pair = new_pair
                    current_start = timestamp
                    current_end = timestamp
                elif current_pair == new_pair:
                    # Same pair, extend the range
                    current_end = timestamp
                else:
                    # Pair changed, save previous range and start new one
                    pair_ranges.append((current_start, current_end, current_pair[0], current_pair[1]))
                    current_pair = new_pair
                    current_start = timestamp
                    current_end = timestamp
    
    # Handle the last range that extends to end of file
    if current_pair is not None:
        pair_ranges.append((current_start, current_end, current_pair[0], current_pair[1]))
    
    return pair_ranges

def annotate_color(color):
    """Annotate color with description."""
    annotations = {
        'C0': 'C0 (a even)',
        'C1': 'C1 (a odd)',
        'C2': 'C2 (b even)',
        'C3': 'C3 (b odd)'
    }
    return annotations.get(color, color)

def annotate_link(link):
    """Annotate link with description."""
    if link == 'R':
        return 'link R (sending)'
    return link

def print_timeline(pair_ranges):
    """Print the timeline of (color, link) pair changes."""
    print("=" * 100)
    print("(COLOR, LINK) PAIR TIMELINE")
    print("=" * 100)
    print(f"{'Range':<8} {'Start':<12} {'End':<12} {'Duration':<12} {'Color':<20} {'Link':<25}")
    print("-" * 100)
    
    if pair_ranges:
        for i, (start, end, color, link) in enumerate(pair_ranges, 1):
            duration = end - start
            annotated_color = annotate_color(color)
            annotated_link = annotate_link(link)
            print(f"{i:<8} @{start:<11} @{end:<11} {duration:<12} {annotated_color:<20} {annotated_link:<25}")
    else:
        print("No (color, link) pairs found.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_color_link_timeline.py <log_file>")
        print("Example: python parse_color_link_timeline.py 4_1_wavelet.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    try:
        pair_ranges = parse_log_file(log_file)
        print_timeline(pair_ranges)
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
