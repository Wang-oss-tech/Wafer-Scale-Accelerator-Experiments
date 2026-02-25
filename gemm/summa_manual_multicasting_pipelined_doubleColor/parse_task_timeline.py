#!/usr/bin/env python3
"""
Parse instruction log file to track task execution ranges.
Reports whenever the active task changes.
"""

import re
import sys

def parse_instr_log(log_file_path):
    """
    Parse the instruction log file and extract timeline information for task execution.
    
    Returns:
        - task_ranges: List of (start_time, end_time, task_id) tuples
    """
    task_ranges = []
    
    # Pattern to match instruction log entries
    # Format: @timestamp P4.1: Id: 15, Instr: 10989, Seq: 0, Pipe: 2, Msg: [IS OP] | 0x01a1: T11 ...
    # We look for the task identifier (T followed by digits)
    pattern = re.compile(r'@(\d+)\s+.*?Msg:\s+\[(?:IS|EX)\s+OP\].*?:\s+(T\d+)\s+')
    
    current_task = None
    current_start = None
    current_end = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                timestamp = int(match.group(1))
                task_id = match.group(2)
                
                if current_task is None:
                    # First task encountered
                    current_task = task_id
                    current_start = timestamp
                    current_end = timestamp
                elif current_task == task_id:
                    # Same task, extend the range
                    current_end = timestamp
                else:
                    # Task changed, save previous range and start new one
                    task_ranges.append((current_start, current_end, current_task))
                    current_task = task_id
                    current_start = timestamp
                    current_end = timestamp
    
    # Handle the last range that extends to end of file
    if current_task is not None:
        task_ranges.append((current_start, current_end, current_task))
    
    return task_ranges

def annotate_task(task_id):
    """
    Annotate task with description based on common task IDs.
    You can customize this based on your specific program.
    """
    annotations = {
        'T0': 'T0 (main thread)',
        'T2': 'T2 (input queue 2)',
        'T3': 'T3 (input queue 3)',
        'T4': 'T4 (input queue 4)',
        'T5': 'T5 (input queue 5)',
        'T11': 'T11 (main/control)',
        'T22': 'T22 (fabric async)',
        'T23': 'T23 (fabric async)',
        'T24': 'T24 (fabric async)',
        'T25': 'T25 (fabric async)',
    }
    return annotations.get(task_id, task_id)

def print_timeline(task_ranges):
    """Print the timeline of task execution changes."""
    print("=" * 100)
    print("TASK EXECUTION TIMELINE")
    print("=" * 100)
    print(f"{'Range':<8} {'Start':<12} {'End':<12} {'Duration':<12} {'Task':<30}")
    print("-" * 100)
    
    if task_ranges:
        for i, (start, end, task_id) in enumerate(task_ranges, 1):
            duration = end - start
            annotated_task = annotate_task(task_id)
            print(f"{i:<8} @{start:<11} @{end:<11} {duration:<12} {annotated_task:<30}")
    else:
        print("No tasks found.")

def print_summary(task_ranges):
    """Print summary statistics for each task."""
    print("\n" + "=" * 100)
    print("TASK EXECUTION SUMMARY")
    print("=" * 100)
    print(f"{'Task':<30} {'Total Cycles':<15} {'Occurrences':<15} {'Avg Duration':<15}")
    print("-" * 100)
    
    # Collect statistics
    task_stats = {}
    for start, end, task_id in task_ranges:
        duration = end - start
        annotated_task = annotate_task(task_id)
        
        if annotated_task not in task_stats:
            task_stats[annotated_task] = {'total': 0, 'count': 0}
        
        task_stats[annotated_task]['total'] += duration
        task_stats[annotated_task]['count'] += 1
    
    # Print statistics sorted by total cycles
    for task, stats in sorted(task_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        total = stats['total']
        count = stats['count']
        avg = total / count if count > 0 else 0
        print(f"{task:<30} {total:<15} {count:<15} {avg:<15.2f}")
    
    # Print overall stats
    if task_ranges:
        total_duration = task_ranges[-1][1] - task_ranges[0][0]
        print("-" * 100)
        print(f"{'Total execution time:':<30} {total_duration} cycles")
        print(f"{'Number of task switches:':<30} {len(task_ranges)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_task_timeline.py <instr_log_file>")
        print("Example: python parse_task_timeline.py 4_1_instr.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    try:
        task_ranges = parse_instr_log(log_file)
        print_timeline(task_ranges)
        print_summary(task_ranges)
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
