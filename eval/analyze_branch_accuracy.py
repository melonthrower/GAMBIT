import json
import os
import argparse
import sys
from collections import defaultdict
from typing import List, Any, Dict, Set

def find_longest_common_prefix(sequences: List[List[Any]]) -> List[Any]:
    if not sequences: return []
    min_len_seq = min(sequences, key=len)
    prefix = []
    for i, item in enumerate(min_len_seq):
        if all(seq[i] == item for seq in sequences):
            prefix.append(item)
        else:
            break
    return prefix

def analyze_branching_accuracy(summary_path: str, annotations_dir: str, verbose: bool):

    summary_path = os.path.expanduser(summary_path)
    annotations_dir = os.path.expanduser(annotations_dir)

    if not os.path.exists(summary_path):
        print(f"Error: no summary file in '{summary_path}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(annotations_dir):
        print(f"Error: Annotation file directory not found in '{annotations_dir}'", file=sys.stderr)
        sys.exit(1)

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    summary_lookup: Dict[str, Dict] = {ep.get("episode_id"): ep for ep in summary_data if ep.get("episode_id")}
    
    first_branch_stats = {'total': 0, 'correct': 0}
    other_branches_stats = {'total': 0, 'correct': 0}

    annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])

    for filename in annotation_files:
        episode_id = os.path.splitext(filename)[0]
        annotation_path = os.path.join(annotations_dir, filename)

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        sequences_dict = annotation_data.get("seq", {})
        if not sequences_dict or len(sequences_dict) < 2:
            continue

        if verbose: print(f"\n[Task {episode_id}]: Identified as a branch instruction task.")
        
        episode_summary = summary_lookup.get(episode_id)
        if not episode_summary:
            if verbose: print(f"  -> Skip (Reason: Evaluation record does not exist).")
            continue

        sequences = list(sequences_dict.values())
        lcp = find_longest_common_prefix(sequences)
        decision_step_id = lcp[-1] if lcp else -1

        frame_to_branch_index: Dict[int, int] = {}
        sorted_seq_keys = sorted(sequences_dict.keys())
        
        for i, seq_key in enumerate(sorted_seq_keys):
            branch_index, seq = i, sequences_dict[seq_key]
            frame_id = -1
            if not lcp:
                if seq: frame_id = seq[0]
            else:
                try:
                    idx = seq.index(decision_step_id)
                    if idx + 1 < len(seq): frame_id = seq[idx + 1]
                except ValueError: pass
            
            if frame_id != -1:
                frame_to_branch_index[frame_id] = branch_index

        if verbose: print(f" - Branch frame -> Branch index mapping: {frame_to_branch_index}")

        executed_steps = episode_summary.get("step_accuracy", {})
        executed_step_ids = {int(k.split('_')[1]) for k in executed_steps.keys()}
        executed_branch_frames = executed_step_ids.intersection(frame_to_branch_index.keys())

        if not executed_branch_frames:
            if verbose: print(f"  -> Result: No valid branch frames were executed.")
            continue

        for frame_id in sorted(list(executed_branch_frames)):
            branch_index = frame_to_branch_index.get(frame_id)
            if branch_index is None: continue

            if branch_index == 0:
                target_stats = first_branch_stats
            else:
                target_stats = other_branches_stats
            
            target_stats['total'] += 1
            
            step_result = executed_steps.get(f"step_{frame_id}")
            if step_result and step_result.get("exact_match", False):
                target_stats['correct'] += 1
                if verbose: print(f"  - Executed branch frame {frame_id} (branch {branch_index}): Correct")
            else:
                if verbose: print(f"  - Executed branch frame {frame_id} (branch {branch_index}): Wrong")

    print("\n" + "=" * 50)
    print("===== Accuracy in different branch=====")

    total_executed = first_branch_stats['total'] + other_branches_stats['total']

    if total_executed == 0:
        print("\nno valid branch step in tasks")
    else:
        total_first = first_branch_stats['total']
        correct_first = first_branch_stats['correct']
        accuracy_first = (correct_first / total_first) * 100 if total_first > 0 else 0
        label_first = "the first branch (seq_0)".ljust(25)
        print(f"{label_first}: accuracy = {accuracy_first:>6.2f}% ({correct_first} / {total_first})")

        total_other = other_branches_stats['total']
        correct_other = other_branches_stats['correct']
        accuracy_other = (correct_other / total_other) * 100 if total_other > 0 else 0
        label_other = "other branch (seq_1+)".ljust(25)
        print(f"{label_other}: accuracy = {accuracy_other:>6.2f}% ({correct_other} / {total_other})")
        
        print("-" * 50)
        
        total_all = total_first + total_other
        correct_all = correct_first + correct_other
        accuracy_all = (correct_all / total_all) * 100 if total_all > 0 else 0
        label_all = "total".ljust(25)
        print(f"{label_all}: accuracy = {accuracy_all:>6.2f}% ({correct_all} / {total_all})")

    print("=" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Divide the branches into two groups: 'the first' and 'the second and above', and count their accuracy rates respectively")
    parser.add_argument("--summary_path", type=str, required=True, help="the path of 'stepwise_accuracy_summary.json'")
    parser.add_argument("--annotations_dir", type=str, required=True, help="the path of origin annotations directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output a detailed analysis process log.")
    
    args = parser.parse_args()
    
    analyze_branching_accuracy(args.summary_path, args.annotations_dir, args.verbose)