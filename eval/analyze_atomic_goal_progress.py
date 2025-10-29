import json
import os
import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Any, Optional

def get_step_to_op_map(atomic_instructions: Dict[str, List[List[int]]]) -> Dict[int, str]:
    mapping = {}
    for op_name, step_lists in atomic_instructions.items():
        for step_list in step_lists:
            for step_id in step_list:
                mapping[int(step_id)] = op_name
    return mapping

def get_ordered_ops_for_seq(seq_steps: List[int], step_to_op_map: Dict[int, str]) -> List[str]:
    ordered_ops = []
    seen_ops = set()
    for step_id in sorted(seq_steps):
        op_name = step_to_op_map.get(int(step_id))
        if op_name and op_name not in seen_ops:
            ordered_ops.append(op_name)
            seen_ops.add(op_name)
    return ordered_ops

def calculate_branch_metrics(
    seq_steps: List[int],
    ordered_ops: List[str],
    atomic_instructions: Dict[str, List[List[int]]],
    step_accuracy: Dict[str, Dict[str, bool]]
) -> Dict[str, float]:
    total_ops_in_branch = len(ordered_ops)
    consecutive_correct_ops = 0
    for op_name in ordered_ops:
        op_steps_in_branch = {
            step for sublist in atomic_instructions[op_name] for step in sublist
        }.intersection(set(seq_steps))
        is_op_correct = True
        if not op_steps_in_branch:
             is_op_correct = True
        else:
            for step_id in op_steps_in_branch:
                step_key = f"step_{step_id}"
                if not step_accuracy.get(step_key, {}).get("exact_match", False):
                    is_op_correct = False
                    break
        if is_op_correct:
            consecutive_correct_ops += 1
        else:
            break
    op_gp_standard = (consecutive_correct_ops / total_ops_in_branch) if total_ops_in_branch > 0 else 0.0

    total_steps_in_branch = len(seq_steps)
    consecutive_correct_steps = 0
    for i in range(consecutive_correct_ops):
        op_name = ordered_ops[i]
        op_steps_in_branch = {
            step for sublist in atomic_instructions[op_name] for step in sublist
        }.intersection(set(seq_steps))
        consecutive_correct_steps += len(op_steps_in_branch)
    step_gp_standard = (consecutive_correct_steps / total_steps_in_branch) if total_steps_in_branch > 0 else 0.0

    longest_step_streak = 0
    current_streak = 0
    if seq_steps:
        for step_id in sorted(seq_steps):
            step_key = f"step_{step_id}"
            if step_accuracy.get(step_key, {}).get("exact_match", False):
                current_streak += 1
            else:
                longest_step_streak = max(longest_step_streak, current_streak)
                current_streak = 0
        longest_step_streak = max(longest_step_streak, current_streak)

    from_start_correct_steps = 0
    if seq_steps:
        for step_id in sorted(seq_steps):
            step_key = f"step_{step_id}"
            if step_accuracy.get(step_key, {}).get("exact_match", False):
                from_start_correct_steps += 1
            else:
                break
    
    initial_step_streak_percentage = (from_start_correct_steps / total_steps_in_branch) if total_steps_in_branch > 0 else 0.0

    return {
        "op_gp_standard": op_gp_standard,
        "step_gp_standard": step_gp_standard,
        "weight_op_count": total_ops_in_branch,
        "weight_step_count": total_steps_in_branch,
        "consecutive_correct_op_count": float(consecutive_correct_ops),
        "longest_step_streak": float(longest_step_streak),
        "from_start_correct_steps": float(from_start_correct_steps),
        "initial_step_streak_percentage": initial_step_streak_percentage,
    }

def analyze_episode(
    annotation_data: Dict[str, Any],
    episode_summary: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    atomic_instructions = annotation_data.get("atomic_instructions", {})
    sequences = annotation_data.get("seq", {})
    if not atomic_instructions:
        return None

    step_accuracy = episode_summary.get("step_accuracy", {})
    step_to_op_map = get_step_to_op_map(atomic_instructions)

    if not sequences:
        all_steps = sorted(list(step_to_op_map.keys()))
        sequences = {"default_seq": all_steps}

    branch_results = []
    for seq_name, seq_steps in sequences.items():
        if not seq_steps: continue
        int_seq_steps = [int(s) for s in seq_steps]
        ordered_ops = get_ordered_ops_for_seq(int_seq_steps, step_to_op_map)
        if not ordered_ops: continue
        branch_metrics = calculate_branch_metrics(int_seq_steps, ordered_ops, atomic_instructions, step_accuracy)
        branch_results.append(branch_metrics)

    if not branch_results:
        return None

    metric_1a = sum(b['op_gp_standard'] for b in branch_results) / len(branch_results)
    total_ops_for_weighting = sum(b['weight_op_count'] for b in branch_results)
    if total_ops_for_weighting > 0:
        numerator_1b = sum(b['op_gp_standard'] * b['weight_op_count'] for b in branch_results)
        metric_1b = numerator_1b / total_ops_for_weighting
    else:
        metric_1b = 0.0

    metric_2a = sum(b['step_gp_standard'] for b in branch_results) / len(branch_results)
    total_steps_for_weighting = sum(b['weight_step_count'] for b in branch_results)
    if total_steps_for_weighting > 0:
        numerator_2b = sum(b['step_gp_standard'] * b['weight_step_count'] for b in branch_results)
        metric_2b = numerator_2b / total_steps_for_weighting
    else:
        metric_2b = 0.0

    metric_3a = sum(b['consecutive_correct_op_count'] for b in branch_results) / len(branch_results)
    if total_ops_for_weighting > 0:
        numerator_3b = sum(b['consecutive_correct_op_count'] * b['weight_op_count'] for b in branch_results)
        metric_3b = numerator_3b / total_ops_for_weighting
    else:
        metric_3b = 0.0

    metric_4a = sum(b['longest_step_streak'] for b in branch_results) / len(branch_results)
    if total_steps_for_weighting > 0:
        numerator_4b = sum(b['longest_step_streak'] * b['weight_step_count'] for b in branch_results)
        metric_4b = numerator_4b / total_steps_for_weighting
    else:
        metric_4b = 0.0

    metric_5a = sum(b['from_start_correct_steps'] for b in branch_results) / len(branch_results)
    if total_steps_for_weighting > 0:
        numerator_5b = sum(b['from_start_correct_steps'] * b['weight_step_count'] for b in branch_results)
        metric_5b = numerator_5b / total_steps_for_weighting
    else:
        metric_5b = 0.0

    metric_6a = sum(b['initial_step_streak_percentage'] for b in branch_results) / len(branch_results)
    if total_steps_for_weighting > 0:
        numerator_6b = sum(b['initial_step_streak_percentage'] * b['weight_step_count'] for b in branch_results)
        metric_6b = numerator_6b / total_steps_for_weighting
    else:
        metric_6b = 0.0

    total_ops_in_episode = len(atomic_instructions)
    total_steps_in_episode = len(annotation_data.get("steps", []))

    return {
        "episode_id": annotation_data["episode_id"],
        "category": annotation_data.get("category", "unknown"),
        "metric_1a": metric_1a, "metric_1b": metric_1b,
        "metric_2a": metric_2a, "metric_2b": metric_2b,
        "metric_3a": metric_3a, "metric_3b": metric_3b,
        "metric_4a": metric_4a, "metric_4b": metric_4b,
        "metric_5a": metric_5a, "metric_5b": metric_5b,
        "metric_6a": metric_6a, "metric_6b": metric_6b,
        "total_ops": total_ops_in_episode,
        "total_steps": total_steps_in_episode,
    }

def create_fixed_bins(
    results_subset: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    bins = {
        "2-5 steps": [],
        "6-8 steps": [],
        "9-12 steps": [],
        "13+ steps": []
    }
    if not results_subset:
        return {}
    for result in results_subset:
        total_steps = result.get('total_steps', 0)
        if 2 <= total_steps <= 5:
            bins["2-5 steps"].append(result)
        elif 6 <= total_steps <= 8:
            bins["6-8 steps"].append(result)
        elif 9 <= total_steps <= 12:
            bins["9-12 steps"].append(result)
        elif total_steps >= 13:
            bins["13+ steps"].append(result)
            
    return {name: group for name, group in bins.items() if group}

def print_report_for_subset(
    results_subset: List[Dict[str, Any]], 
    title: str
):
    print("\n" + "#"*180)
    print(f"##### {title.center(170)} #####")
    print("#"*180)

    if not results_subset:
        print("no data in this category。".center(180))
        return

    table_data = []

    def calculate_final_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results: return {"1a": 0.0, "1b": 0.0, "2a": 0.0, "2b": 0.0, "3a": 0.0, "3b": 0.0, "4a": 0.0, "4b": 0.0, "5a": 0.0, "5b": 0.0, "6a": 0.0, "6b": 0.0, "count": 0}
        count = len(results)
        
        # 'a' series simple averages
        final_1a = sum(r['metric_1a'] for r in results) / count
        final_2a = sum(r['metric_2a'] for r in results) / count
        final_3a = sum(r['metric_3a'] for r in results) / count
        final_4a = sum(r.get('metric_4a', 0.0) for r in results) / count
        final_5a = sum(r.get('metric_5a', 0.0) for r in results) / count
        final_6a = sum(r.get('metric_6a', 0.0) for r in results) / count
        
        # 'b' series weighted by total_ops
        total_ops_dataset = sum(r['total_ops'] for r in results)
        num_1b = sum(r['metric_1b'] * r['total_ops'] for r in results)
        final_1b = num_1b / total_ops_dataset if total_ops_dataset > 0 else 0.0
        num_3b = sum(r.get('metric_3b', r['metric_3a']) * r['total_ops'] for r in results)
        final_3b = num_3b / total_ops_dataset if total_ops_dataset > 0 else 0.0

        # 'b' series weighted by total_steps
        total_steps_dataset = sum(r['total_steps'] for r in results)
        num_2b = sum(r['metric_2b'] * r['total_steps'] for r in results)
        final_2b = num_2b / total_steps_dataset if total_steps_dataset > 0 else 0.0
        num_4b = sum(r.get('metric_4b', 0.0) * r['total_steps'] for r in results)
        final_4b = num_4b / total_steps_dataset if total_steps_dataset > 0 else 0.0
        num_5b = sum(r.get('metric_5b', 0.0) * r['total_steps'] for r in results)
        final_5b = num_5b / total_steps_dataset if total_steps_dataset > 0 else 0.0
        num_6b = sum(r.get('metric_6b', 0.0) * r['total_steps'] for r in results)
        final_6b = num_6b / total_steps_dataset if total_steps_dataset > 0 else 0.0
        
        return {
            "1a": final_1a, "1b": final_1b, "2a": final_2a, "2b": final_2b,
            "3a": final_3a, "3b": final_3b, "4a": final_4a, "4b": final_4b,
            "5a": final_5a, "5b": final_5b, "6a": final_6a, "6b": final_6b, "count": count
        }

    overall_metrics = calculate_final_metrics(results_subset)
    overall_metrics['name'] = "Overall"
    table_data.append(overall_metrics)

    grouped_results = create_fixed_bins(results_subset)
    if grouped_results:
        group_order = ["2-5 steps", "6-8 steps", "9-12 steps", "13+ steps"]
        sorted_group_names = [name for name in group_order if name in grouped_results]
        
        for name in sorted_group_names:
            group_data = grouped_results[name]
            metrics = calculate_final_metrics(group_data)
            metrics['name'] = name
            table_data.append(metrics)
    else:
        print(f"no data in these groups".center(180))


    COL_GROUP, COL_TASKS, COL_METRIC_PCT, COL_METRIC_ABS = 22, 7, 10, 11
    header = (
        f"{'Group':<{COL_GROUP}} | {'Tasks':>{COL_TASKS}} | "
        f"{'1a (%)':>{COL_METRIC_PCT}} {'1b (%)':>{COL_METRIC_PCT}} | "
        f"{'2a (%)':>{COL_METRIC_PCT}} {'2b (%)':>{COL_METRIC_PCT}} | "
        f"{'3a (Abs)':>{COL_METRIC_ABS}} {'3b (Abs)':>{COL_METRIC_ABS}} | "
        f"{'4a (Abs)':>{COL_METRIC_ABS}} {'4b (Abs)':>{COL_METRIC_ABS}} | "
        f"{'5a (Abs)':>{COL_METRIC_ABS}} {'5b (Abs)':>{COL_METRIC_ABS}} | "
        f"{'6a (%)':>{COL_METRIC_PCT}} {'6b (%)':>{COL_METRIC_PCT}}"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(table_data):
        row_str = (
            f"{row['name']:<{COL_GROUP}} | {row['count']:>{COL_TASKS}} | "
            f"{row['1a'] * 100:>{COL_METRIC_PCT}.2f} {row['1b'] * 100:>{COL_METRIC_PCT}.2f} | "
            f"{row['2a'] * 100:>{COL_METRIC_PCT}.2f} {row['2b'] * 100:>{COL_METRIC_PCT}.2f} | "
            f"{row['3a']:>{COL_METRIC_ABS}.2f} {row['3b']:>{COL_METRIC_ABS}.2f} | "
            f"{row['4a']:>{COL_METRIC_ABS}.2f} {row['4b']:>{COL_METRIC_ABS}.2f} | "
            f"{row['5a']:>{COL_METRIC_ABS}.2f} {row['5b']:>{COL_METRIC_ABS}.2f} | "
            f"{row['6a'] * 100:>{COL_METRIC_PCT}.2f} {row['6b'] * 100:>{COL_METRIC_PCT}.2f}"
        )
        print(row_str)
        if i == 0 and len(table_data) > 1:
            print("-" * len(header))

def group_and_print_results(all_results: List[Dict[str, Any]]):
    print_report_for_subset(
        all_results, 
        "Overall Results (All Categories)"
    )

    print("\n" + "="*180)
    print("Analysis Finished".center(180))
    print("="*180)

def main():
    parser = argparse.ArgumentParser(description="Calculate advanced, atomic-instruction-based Goal Progress metrics.")
    parser.add_argument(
        "--summary_path", 
        type=str, 
        required=True,
        help="Path to the 'stepwise_accuracy_summary.json' file."
    )
    parser.add_argument(
        "--annotations_dir", 
        type=str, 
        required=True,
        help="Path to the directory containing all original annotation JSON files."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.summary_path):
        print(f"Error: Summary file not found at '{args.summary_path}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.annotations_dir):
        print(f"Error: Annotations directory not found at '{args.annotations_dir}'", file=sys.stderr)
        sys.exit(1)

    with open(args.summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    summary_lookup: Dict[str, Dict] = {ep["episode_id"]: ep for ep in summary_data if "episode_id" in ep}
    
    all_episode_results = []
    annotation_files = sorted([f for f in os.listdir(args.annotations_dir) if f.endswith('.json')])

    processed_count = 0
    skipped_no_summary = 0

    for filename in annotation_files:
        episode_id = os.path.splitext(filename)[0]
        annotation_path = os.path.join(args.annotations_dir, filename)

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
        except Exception:
            continue
            
        if "atomic_instructions" not in annotation_data or not annotation_data["atomic_instructions"]:
            continue

        if episode_id not in summary_lookup:
            skipped_no_summary += 1
            continue
            
        episode_summary = summary_lookup[episode_id]
        
        results = analyze_episode(annotation_data, episode_summary)
        if results:
            all_episode_results.append(results)
            processed_count += 1

    print(f"Analysis Complete. Processed {processed_count} episodes with atomic instructions.")
    if skipped_no_summary > 0:
        print(f"Warning: Skipped {skipped_no_summary} episodes that had atomic instructions but were not found in the summary file.")

    if all_episode_results:
        group_and_print_results(all_episode_results)
    else:
        print("No episodes with atomic instructions were found to analyze.")

if __name__ == "__main__":
    main()