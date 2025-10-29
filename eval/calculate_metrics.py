import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Union

def find_longest_common_prefix(sequences: List[List[int]]) -> List[int]:
    if not sequences: return []
    sorted_sequences = sorted(sequences, key=len)
    shortest_seq = sorted_sequences[0]
    prefix = []
    for i, step in enumerate(shortest_seq):
        if all(seq[i] == step for seq in sorted_sequences[1:]):
            prefix.append(step)
        else:
            break
    return prefix

def map_seq_to_atomic_actions(sequence: List[int], atomic_instructions: Dict) -> List[Dict[str, Any]]:
    if not atomic_instructions:
        return [{"name": "unnamed_atomic_action", "steps": sequence}]
    all_actions = []
    for name, groups in atomic_instructions.items():
        for group in groups:
            if group:
                all_actions.append({"name": name, "steps": group})
    all_actions.sort(key=lambda x: len(x["steps"]), reverse=True)
    ordered_actions = []
    unmatched_steps = []
    i = 0
    while i < len(sequence):
        best_match = None
        for action_def in all_actions:
            action_steps = action_def["steps"]
            len_action = len(action_steps)
            if sequence[i : i + len_action] == action_steps:
                best_match = action_def
                break
        if best_match:
            if unmatched_steps:
                ordered_actions.append({"name": "unmatched_action", "steps": unmatched_steps})
                unmatched_steps = []
            ordered_actions.append(best_match)
            i += len(best_match["steps"])
        else:
            unmatched_steps.append(sequence[i])
            i += 1
    if unmatched_steps:
        ordered_actions.append({"name": "unmatched_action", "steps": unmatched_steps})
    return ordered_actions


def calculate_path_metrics(episode_summary: Dict, annotations_dir: str) -> Union[Dict, None]:
    episode_id = episode_summary.get('episode_id')
    is_branching_path = '_' in str(episode_id)
    if is_branching_path:
        original_id, branch_suffix_str = str(episode_id).rsplit('_', 1)
        try:
            branch_suffix = int(branch_suffix_str)
            seq_key = f"seq_{branch_suffix}"
        except ValueError: return None
    else:
        original_id = str(episode_id)
        seq_key = "seq_0"
    annotation_file = Path(annotations_dir) / f"{original_id}.json"
    if not annotation_file.exists(): return None
    with open(annotation_file, 'r', encoding='utf-8') as f: annotation_data = json.load(f)
    category = annotation_data.get('category', 'unknown')
    seq_map = annotation_data.get('seq', {})
    path_steps = seq_map.get(seq_key)
    if not path_steps: return None
    step_accuracies = episode_summary.get('step_accuracy', {})
    correct_steps = {}
    for step_key, accuracy_info in step_accuracies.items():
        try:
            relative_idx = int(step_key.split('_')[1])
            if relative_idx < len(path_steps):
                absolute_step_id = path_steps[relative_idx]
                correct_steps[absolute_step_id] = accuracy_info.get('exact_match', False)
        except (ValueError, IndexError, KeyError): continue
    
    atomic_instructions = annotation_data.get('atomic_instructions', {})
    atomic_actions_for_path = map_seq_to_atomic_actions(path_steps, atomic_instructions)
    
    continuous_action_count = 0
    continuous_step_count_from_ops = 0
    for action in atomic_actions_for_path:
        action_steps = action['steps']
        if all(correct_steps.get(step, False) for step in action_steps):
            continuous_action_count += 1
            continuous_step_count_from_ops += len(action_steps)
        else: 
            break

    initial_step_streak = 0
    for step in path_steps:
        if correct_steps.get(step, False):
            initial_step_streak += 1
        else:
            break

    max_continuous_correct_steps = 0
    current_streak = 0
    for step in path_steps:
        if correct_steps.get(step, False):
            current_streak += 1
        else:
            max_continuous_correct_steps = max(max_continuous_correct_steps, current_streak)
            current_streak = 0
    max_continuous_correct_steps = max(max_continuous_correct_steps, current_streak)
    
    return {
        "original_episode_id": original_id, "episode_id": episode_id, "category": category, 
        "continuous_action_count": continuous_action_count,
        "total_actions_in_path": len(atomic_actions_for_path), 
        "continuous_step_count_from_ops": continuous_step_count_from_ops,
        "initial_step_streak": initial_step_streak,
        "max_continuous_correct_steps": max_continuous_correct_steps,
        "total_steps_in_path": len(path_steps), "path_steps": path_steps
    }

def analyze_branching_decisions(summary_data: List[Dict], annotations_dir: str):
    branching_tasks = defaultdict(list)
    for episode_summary in summary_data:
        episode_id = episode_summary.get("episode_id")
        if episode_id and '_' in str(episode_id):
            original_id = str(episode_id).split('_')[0]
            branching_tasks[original_id].append(episode_summary)
    first_branch_stats = {'total': 0, 'correct': 0}
    other_branches_stats = {'total': 0, 'correct': 0}
    for original_id, summaries in branching_tasks.items():
        annotation_file = Path(annotations_dir) / f"{original_id}.json"
        if not annotation_file.exists(): continue
        with open(annotation_file, 'r', encoding='utf-8') as f: annotation_data = json.load(f)
        sequences_dict = annotation_data.get("seq", {})
        if len(sequences_dict) < 2: continue
        sequences = list(sequences_dict.values())
        lcp = find_longest_common_prefix(sequences)
        for summary in summaries:
            episode_id = summary.get("episode_id")
            if not episode_id: continue
            try:
                branch_suffix = int(str(episode_id).split('_')[-1])
                seq_key = f"seq_{branch_suffix}"
            except (IndexError, ValueError): continue
            sequence_path_abs = sequences_dict.get(seq_key)
            if not sequence_path_abs: continue
            if not lcp:
                decision_frame_relative_idx = 0
            else:
                pre_decision_step = lcp[-1]
                try:
                    pre_decision_idx_in_path = sequence_path_abs.index(pre_decision_step)
                    decision_frame_relative_idx = pre_decision_idx_in_path + 1
                except ValueError: continue
            if decision_frame_relative_idx >= len(sequence_path_abs): continue
            step_key_relative = f"step_{decision_frame_relative_idx}"
            executed_steps = summary.get("step_accuracy", {})
            if step_key_relative in executed_steps:
                stats_bucket = first_branch_stats if branch_suffix == 0 else other_branches_stats
                stats_bucket['total'] += 1
                step_result = executed_steps[step_key_relative]
                if step_result and step_result.get("exact_match", False):
                    stats_bucket['correct'] += 1
    print("\n" + "="*80)
    print(" " * 28 + "branch step accuracy report")
    print("="*80)
    total_decisions = first_branch_stats['total'] + other_branches_stats['total']
    if total_decisions == 0:
        print("no branch step detacted")
    else:
        total_first = first_branch_stats['total']
        correct_first = first_branch_stats['correct']
        acc_first = (correct_first / total_first) * 100 if total_first > 0 else 0.0
        print(f"{'The first branch (seq_0)':<30}: accuracy = {acc_first:6.2f}% ({correct_first} / {total_first})")
        total_other = other_branches_stats['total']
        correct_other = other_branches_stats['correct']
        acc_other = (correct_other / total_other) * 100 if total_other > 0 else 0.0
        print(f"{'other branch (seq_1+)':<30}: accuracy = {acc_other:6.2f}% ({correct_other} / {total_other})")
        print("-" * 80)
        total_all = total_first + total_other
        correct_all = correct_first + correct_other
        acc_all = (correct_all / total_all) * 100 if total_all > 0 else 0.0
        print(f"{'total':<30}: accuracy = {acc_all:6.2f}% ({correct_all} / {total_all})")
    print("="*80)

def generate_advanced_report(final_episode_metrics: List[Dict], title: str):
    print("\n" + "="*180)
    print(f"{'===== Advanced Goal Progress Metrics based on Atomic Instructions =====':^180}")
    print("="*180)
    print("Indicator Explanation:")
    print(" 1.a/b. Atomic Op GP : Proportion of atomic operations completed from scratch (simple/Op weighted average)")
    print(" 2.a/b. Step-level GP : Proportion of steps completed from scratch (simple/Step weighted average) [Based on atomic operations]")
    print(" 3.a/b. Atomic Op Count : Absolute number of atomic operations completed from scratch (simple/Op weighted average)")
    print(" 4.a/b. Longest Step Streak : Average of longest consecutive correct steps within branches (simple/Step weighted average) [Based on steps]")
    print(" 5.a/b. Initial Step Streak   : The average value of [the number of consecutive correct steps starting from the beginning] within the branch (simple/Step weighted average) [based on steps]")    
    print(" 6.a/b. Initial Step Streak Pct : The average value of [the percentage of consecutive correct steps starting from the beginning] within the branch (simple/Step weighted average) [based on steps]")
    print("\n" + "#"*180)
    print(f"#####{title:^170}#####")
    print("#"*180)
    headers = (
        f"{'Group':<22} | {'Tasks':>7} | "
        f"{'1a (%)':>10} {'1b (%)':>10} | "
        f"{'2a (%)':>10} {'2b (%)':>10} | "
        f"{'3a (Abs)':>10} {'3b (Abs)':>10} | "
        f"{'4a (Abs)':>10} {'4b (Abs)':>10} | "
        f"{'5a (Abs)':>10} {'5b (Abs)':>10} | "
        f"{'6a (%)':>10} {'6b (%)':>10}"
    )
    print(headers)
    print("-" * len(headers))
    def _calculate_and_print_row(group_name: str, metrics_group: List[Dict]):
        num_tasks = len(metrics_group)
        if num_tasks == 0: return
        m1a = sum(m['atomic_op_gp_simple'] for m in metrics_group) / num_tasks * 100
        m2a = sum(m['step_level_gp_simple'] for m in metrics_group) / num_tasks * 100
        m3a = sum(m['atomic_op_count_simple'] for m in metrics_group) / num_tasks
        m4a = sum(m['longest_streak_simple'] for m in metrics_group) / num_tasks
        m5a = sum(m['initial_streak_simple'] for m in metrics_group) / num_tasks
        m6a = sum(m['initial_streak_pct_simple'] for m in metrics_group) / num_tasks * 100
        total_cont_ops = sum(m['sum_continuous_actions'] for m in metrics_group)
        total_ops = sum(m['sum_total_actions'] for m in metrics_group)
        total_cont_steps_from_ops = sum(m['sum_continuous_steps_from_ops'] for m in metrics_group)
        total_steps = sum(m['sum_total_steps'] for m in metrics_group)
        m1b = (total_cont_ops / total_ops) * 100 if total_ops > 0 else 0
        m2b = (total_cont_steps_from_ops / total_steps) * 100 if total_steps > 0 else 0 # 使用基于Op的计数值
        total_weight_ops = total_ops
        total_weight_steps = total_steps
        sum_weighted_3b = sum(m['atomic_op_count_weighted'] * m['sum_total_actions'] for m in metrics_group)
        m3b = sum_weighted_3b / total_weight_ops if total_weight_ops > 0 else 0
        sum_weighted_4b = sum(m['longest_streak_weighted'] * m['sum_total_steps'] for m in metrics_group)
        m4b = sum_weighted_4b / total_weight_steps if total_weight_steps > 0 else 0
        sum_weighted_5b = sum(m['initial_streak_weighted'] * m['sum_total_steps'] for m in metrics_group)
        m5b = sum_weighted_5b / total_weight_steps if total_weight_steps > 0 else 0
        sum_weighted_6b = sum(m['initial_streak_pct_weighted'] * m['sum_total_steps'] for m in metrics_group)
        m6b = sum_weighted_6b / total_weight_steps * 100 if total_weight_steps > 0 else 0
        row = (
            f"{group_name:<22} | {num_tasks:>7} | "
            f"{m1a:>10.2f} {m1b:>10.2f} | "
            f"{m2a:>10.2f} {m2b:>10.2f} | "
            f"{m3a:>10.2f} {m3b:>10.2f} | "
            f"{m4a:>10.2f} {m4b:>10.2f} | "
            f"{m5a:>10.2f} {m5b:>10.2f} | "
            f"{m6a:>10.2f} {m6b:>10.2f}"
        )
        print(row)
    _calculate_and_print_row("Overall", final_episode_metrics)
    print("-" * len(headers))

    buckets_def = { 
        "2-5 steps": (2, 5), 
        "6-8 steps": (6, 8), 
        "9-12 steps": (9, 12), 
        "13+ steps": (13, float('inf')) 
    }
    
    for name, (min_len, max_len) in buckets_def.items():
        bucket_metrics = [m for m in final_episode_metrics if min_len <= m['task_length'] <= max_len]
        _calculate_and_print_row(name, bucket_metrics)
    print("\n" + "="*180)
    print(f"{'Analysis Finished':^180}")
    print("="*180)

def analyze_by_category(summary_data: List[Dict], annotations_dir: str):
    print("\n" + "="*145)
    print(" " * 53 + "report grouped by category")
    print("="*145)
    
    category_map, seq_map_cache, gt_action_map = {}, {}, {}
    episode_task_lengths = {}
    for f in Path(annotations_dir).glob('*.json'):
        episode_id = f.stem
        gt_action_map[episode_id] = {}
        try:
            with open(f, 'r', encoding='utf-8') as ann_file:
                data = json.load(ann_file)
                category_map[episode_id] = data.get('category', 'unknown')
                seqs = data.get('seq', {})
                seq_map_cache[episode_id] = seqs
                
                unique_steps = set()
                if seqs:
                    for s in seqs.values():
                        unique_steps.update(s)
                episode_task_lengths[episode_id] = len(unique_steps)

                for i, step_data in enumerate(data.get('steps', [])):
                    action = step_data.get('actions', [{}])[0]
                    action_type = action.get('action_type', 'unknown').lower()
                    gt_action_map[episode_id][i] = action_type
        except Exception: continue
        
    episode_branch_results_strict = defaultdict(lambda: {'total_branches_run': 0, 'successful_branches': 0, 'category': 'unknown'})
    episode_branch_results_no_complete = defaultdict(lambda: {'total_branches_run': 0, 'successful_branches': 0, 'category': 'unknown'})
    episode_branch_gp = defaultdict(list)
    counted_steps_for_category = set()
    category_step_stats = defaultdict(lambda: {'total_steps': 0, 'type_correct': 0, 'exact_correct': 0})
    
    for episode_summary in summary_data:
        episode_id_raw = episode_summary.get('episode_id')
        if not episode_id_raw: continue
        original_id = str(episode_id_raw).split('_')[0]
        seq_key = f"seq_{str(episode_id_raw).split('_')[1]}" if '_' in str(episode_id_raw) else "seq_0"
        category = category_map.get(original_id, 'unknown')
        step_accuracy = episode_summary.get('step_accuracy', {})
        if not step_accuracy: continue
        path_steps_absolute = seq_map_cache.get(original_id, {}).get(seq_key)
        if not path_steps_absolute: continue
        
        branch_is_success_strict = all(v.get('exact_match', False) for v in step_accuracy.values())
        relevant_steps_for_success = []
        for step_key, step_result in step_accuracy.items():
            try:
                relative_idx = int(step_key.split('_')[1])
                if relative_idx >= len(path_steps_absolute): continue
                absolute_step_id = path_steps_absolute[relative_idx]
                action_type = gt_action_map.get(original_id, {}).get(absolute_step_id, 'unknown')
                if action_type != 'complete':
                    relevant_steps_for_success.append(step_result.get('exact_match', False))
            except (ValueError, IndexError): continue
        branch_is_success_no_complete = all(relevant_steps_for_success)
        
        total_steps_in_branch = len(path_steps_absolute)
        continuous_correct_steps = 0
        if total_steps_in_branch > 0:
            for i in range(total_steps_in_branch):
                step_key_check = f"step_{i}"
                if step_accuracy.get(step_key_check, {}).get('exact_match', False): continuous_correct_steps += 1
                else: break
            gp_for_branch = continuous_correct_steps / total_steps_in_branch
        else: gp_for_branch = 0.0
        episode_branch_gp[original_id].append(gp_for_branch)

        results_strict = episode_branch_results_strict[original_id]
        results_strict['total_branches_run'] += 1
        if branch_is_success_strict: results_strict['successful_branches'] += 1
        results_strict['category'] = category

        results_no_complete = episode_branch_results_no_complete[original_id]
        results_no_complete['total_branches_run'] += 1
        if branch_is_success_no_complete: results_no_complete['successful_branches'] += 1
        results_no_complete['category'] = category

        for step_key, step_result in step_accuracy.items():
            try:
                relative_idx = int(step_key.split('_')[1])
                if relative_idx >= len(path_steps_absolute): continue
                absolute_step_id = path_steps_absolute[relative_idx]
            except (ValueError, IndexError): continue
            step_uid = (original_id, absolute_step_id)
            if step_uid in counted_steps_for_category: continue
            counted_steps_for_category.add(step_uid)
            stats = category_step_stats[category]
            stats['total_steps'] += 1
            if step_result.get('type_match', False): stats['type_correct'] += 1
            if step_result.get('exact_match', False): stats['exact_correct'] += 1

    category_final_stats = defaultdict(lambda: {'total': 0, 'strict_success': 0, 'no_complete_success': 0, 'total_gp': 0.0})
    for original_id, results_strict in episode_branch_results_strict.items():
        category = results_strict['category']
        stats = category_final_stats[category]
        stats['total'] += 1
        if results_strict['total_branches_run'] > 0 and results_strict['total_branches_run'] == results_strict['successful_branches']:
            stats['strict_success'] += 1
        results_no_complete = episode_branch_results_no_complete[original_id]
        if results_no_complete['total_branches_run'] > 0 and results_no_complete['total_branches_run'] == results_no_complete['successful_branches']:
            stats['no_complete_success'] += 1
        gp_list = episode_branch_gp.get(original_id, [0.0])
        avg_gp_for_task = sum(gp_list) / len(gp_list) if gp_list else 0.0
        stats['total_gp'] += avg_gp_for_task
        
    print(f"{'Category':<20} | {'Strict SR':<25} | {'SR (no COMPLETE)':<25} | {'Goal Progress':<20} | {'Type Acc':<20} | {'Exact Acc':<20}")
    print("-" * 145)
    overall_stats = defaultdict(float)
    sorted_categories = sorted(category_final_stats.keys())
    for category in sorted_categories:
        final_stats = category_final_stats[category]
        step_stats = category_step_stats[category]
        overall_stats['total'] += final_stats['total']
        overall_stats['strict_success'] += final_stats['strict_success']
        overall_stats['no_complete_success'] += final_stats['no_complete_success']
        overall_stats['total_gp'] += final_stats['total_gp']
        overall_stats['total_steps'] += step_stats['total_steps']
        overall_stats['type_correct'] += step_stats['type_correct']
        overall_stats['exact_correct'] += step_stats['exact_correct']
    for category in sorted_categories:
        final_stats = category_final_stats[category]
        step_stats = category_step_stats[category]
        total = final_stats['total']
        strict_sr = (final_stats['strict_success'] / total) * 100 if total > 0 else 0
        no_complete_sr = (final_stats['no_complete_success'] / total) * 100 if total > 0 else 0
        avg_gp = (final_stats['total_gp'] / total) * 100 if total > 0 else 0.0
        total_s = step_stats['total_steps']
        type_acc = (step_stats['type_correct'] / total_s) * 100 if total_s > 0 else 0
        exact_acc = (step_stats['exact_correct'] / total_s) * 100 if total_s > 0 else 0
        strict_sr_str = f"{strict_sr:.2f}% ({int(final_stats['strict_success'])}/{total})"
        no_complete_sr_str = f"{no_complete_sr:.2f}% ({int(final_stats['no_complete_success'])}/{total})"
        gp_str = f"{avg_gp:.2f}%"
        type_acc_str = f"{type_acc:.2f}% ({step_stats['type_correct']}/{total_s})"
        exact_acc_str = f"{exact_acc:.2f}% ({step_stats['exact_correct']}/{total_s})"
        print(f"{category:<20} | {strict_sr_str:<25} | {no_complete_sr_str:<25} | {gp_str:<20} | {type_acc_str:<20} | {exact_acc_str:<20}")
    
    print("-" * 145)
    total_overall = int(overall_stats['total'])
    total_strict_sr = (overall_stats['strict_success'] / total_overall) * 100 if total_overall > 0 else 0
    total_no_complete_sr = (overall_stats['no_complete_success'] / total_overall) * 100 if total_overall > 0 else 0
    total_avg_gp = (overall_stats['total_gp'] / total_overall) * 100 if total_overall > 0 else 0.0
    total_type_acc = (overall_stats['type_correct'] / overall_stats['total_steps']) * 100 if overall_stats['total_steps'] > 0 else 0
    total_exact_acc = (overall_stats['exact_correct'] / overall_stats['total_steps']) * 100 if overall_stats['total_steps'] > 0 else 0
    total_strict_sr_str = f"{total_strict_sr:.2f}% ({int(overall_stats['strict_success'])}/{total_overall})"
    total_no_complete_sr_str = f"{total_no_complete_sr:.2f}% ({int(overall_stats['no_complete_success'])}/{total_overall})"
    total_gp_str = f"{total_avg_gp:.2f}%"
    total_type_acc_str = f"{total_type_acc:.2f}% ({int(overall_stats['type_correct'])}/{int(overall_stats['total_steps'])})"
    total_exact_acc_str = f"{total_exact_acc:.2f}% ({int(overall_stats['exact_correct'])}/{int(overall_stats['total_steps'])})"
    print(f"{'Overall':<20} | {total_strict_sr_str:<25} | {total_no_complete_sr_str:<25} | {total_gp_str:<20} | {total_type_acc_str:<20} | {total_exact_acc_str:<20}")
    print("="*145)
    print("Note:")    
    print("  - Strict SR: Strict Success Rate. For a task to be considered successful, all its executed branches must be 100% correct.")    
    print("  - SR (no COMPLETE): Success rate after excluding the 'COMPLETE' action. The calculation logic is the same as above, but the result of the 'COMPLETE' step is ignored when determining whether a branch is successful.")    
    print("  - Goal Progress: The percentage of consecutive correct steps from the start of the task relative to the total number of steps. For multi-branch tasks, this value is the average of this indicator across all executed branches.")

    def analyze_and_report_by_step_buckets():
        print("\n" + "="*145)
        print(" " * 53 + "Report Classified by the Number of Task Steps")
        print("="*145)
        
        buckets_def = {
            "2-5 steps": (2, 5), 
            "6-8 steps": (6, 8), 
            "9-12 steps": (9, 12),
            "13+ steps": (13, float('inf'))
        }

        bucketed_tasks = defaultdict(list)
        all_evaluated_task_ids = set(episode_branch_results_strict.keys())
        for task_id in all_evaluated_task_ids:
            task_len = episode_task_lengths.get(task_id, 0)
            for name, (min_len, max_len) in buckets_def.items():
                if min_len <= task_len <= max_len:
                    bucketed_tasks[name].append(task_id)
                    break
        
        headers = f"{'Step Range':<20} | {'Tasks':>7} | {'Strict SR':<25} | {'SR (no COMPLETE)':<25} | {'Goal Progress':<20} | {'Type Acc':<20} | {'Exact Acc':<20}"
        print(headers)
        print("-" * len(headers))

        for bucket_name, task_ids in sorted(bucketed_tasks.items(), key=lambda x: list(buckets_def.keys()).index(x[0])):
            if not task_ids: continue
            
            bucket_stats = defaultdict(float)
            bucket_step_stats = defaultdict(int)
            num_tasks_in_bucket = len(task_ids)

            for task_id in task_ids:
                strict_res = episode_branch_results_strict[task_id]
                if strict_res['total_branches_run'] > 0 and strict_res['total_branches_run'] == strict_res['successful_branches']:
                    bucket_stats['strict_success'] += 1
                
                no_complete_res = episode_branch_results_no_complete[task_id]
                if no_complete_res['total_branches_run'] > 0 and no_complete_res['total_branches_run'] == no_complete_res['successful_branches']:
                    bucket_stats['no_complete_success'] += 1
                
                gp_list = episode_branch_gp.get(task_id, [0.0])
                avg_gp_for_task = sum(gp_list) / len(gp_list) if gp_list else 0.0
                bucket_stats['total_gp'] += avg_gp_for_task

            counted_steps_for_bucket = set()
            for episode_summary in summary_data:
                episode_id_raw = episode_summary.get('episode_id')
                if not episode_id_raw: continue
                original_id = str(episode_id_raw).split('_')[0]
                if original_id not in task_ids: continue
                
                seq_key = f"seq_{str(episode_id_raw).split('_')[1]}" if '_' in str(episode_id_raw) else "seq_0"
                step_accuracy = episode_summary.get('step_accuracy', {})
                path_steps_absolute = seq_map_cache.get(original_id, {}).get(seq_key)
                if not path_steps_absolute: continue

                for step_key, step_result in step_accuracy.items():
                    try:
                        relative_idx = int(step_key.split('_')[1])
                        if relative_idx >= len(path_steps_absolute): continue
                        absolute_step_id = path_steps_absolute[relative_idx]
                    except (ValueError, IndexError): continue
                    
                    step_uid = (original_id, absolute_step_id)
                    if step_uid in counted_steps_for_bucket: continue
                    counted_steps_for_bucket.add(step_uid)
                    
                    bucket_step_stats['total_steps'] += 1
                    if step_result.get('type_match', False): bucket_step_stats['type_correct'] += 1
                    if step_result.get('exact_match', False): bucket_step_stats['exact_correct'] += 1

            total = num_tasks_in_bucket
            strict_sr = (bucket_stats['strict_success'] / total) * 100 if total > 0 else 0
            no_complete_sr = (bucket_stats['no_complete_success'] / total) * 100 if total > 0 else 0
            avg_gp = (bucket_stats['total_gp'] / total) * 100 if total > 0 else 0.0
            
            total_s = bucket_step_stats['total_steps']
            type_acc = (bucket_step_stats['type_correct'] / total_s) * 100 if total_s > 0 else 0
            exact_acc = (bucket_step_stats['exact_correct'] / total_s) * 100 if total_s > 0 else 0

            strict_sr_str = f"{strict_sr:.2f}% ({int(bucket_stats['strict_success'])}/{total})"
            no_complete_sr_str = f"{no_complete_sr:.2f}% ({int(bucket_stats['no_complete_success'])}/{total})"
            gp_str = f"{avg_gp:.2f}%"
            type_acc_str = f"{type_acc:.2f}% ({bucket_step_stats['type_correct']}/{total_s})"
            exact_acc_str = f"{exact_acc:.2f}% ({bucket_step_stats['exact_correct']}/{total_s})"

            print(f"{bucket_name:<20} | {total:>7} | {strict_sr_str:<25} | {no_complete_sr_str:<25} | {gp_str:<20} | {type_acc_str:<20} | {exact_acc_str:<20}")
        
        print("="*145)
    
    analyze_and_report_by_step_buckets()

def analyze_by_action_type(summary_data: List[Dict], annotations_dir: str):
    print("\n" + "="*80)
    print(" " * 25 + "Report Classified by Ground Truth Action Types")
    print("="*80)
    gt_action_map = {}
    seq_map_cache = {}
    for f in Path(annotations_dir).glob('*.json'):
        episode_id = f.stem
        gt_action_map[episode_id] = {}
        try:
            with open(f, 'r', encoding='utf-8') as ann_file:
                data = json.load(ann_file)
                seq_map_cache[episode_id] = data.get('seq', {})
                for i, step_data in enumerate(data.get('steps', [])):
                    action = step_data.get('actions', [{}])[0]
                    action_type = action.get('action_type', 'unknown')
                    gt_action_map[episode_id][i] = action_type
        except Exception: continue
    action_type_stats = defaultdict(lambda: {'total_steps': 0, 'type_correct': 0, 'exact_correct': 0})
    counted_steps_for_action_type = set()
    for episode_summary in summary_data:
        episode_id_raw = episode_summary.get('episode_id')
        if not episode_id_raw: continue
        original_id = str(episode_id_raw).split('_')[0]
        seq_key = f"seq_{str(episode_id_raw).split('_')[1]}" if '_' in str(episode_id_raw) else "seq_0"
        step_accuracy = episode_summary.get('step_accuracy', {})
        path_steps_absolute = seq_map_cache.get(original_id, {}).get(seq_key)
        if not path_steps_absolute: continue
        for step_key, step_result in step_accuracy.items():
            try:
                relative_idx = int(step_key.split('_')[1])
                if relative_idx >= len(path_steps_absolute): continue
                absolute_step_id = path_steps_absolute[relative_idx]
            except (ValueError, IndexError): continue
            step_uid = (original_id, absolute_step_id)
            if step_uid in counted_steps_for_action_type: continue
            counted_steps_for_action_type.add(step_uid)
            gt_action = gt_action_map.get(original_id, {}).get(absolute_step_id, 'unknown')
            stats = action_type_stats[gt_action]
            stats['total_steps'] += 1
            if step_result.get('type_match', False): stats['type_correct'] += 1
            if step_result.get('exact_match', False): stats['exact_correct'] += 1
    print(f"{'Action Type':<20} | {'Type Acc':<25} | {'Exact Acc':<25}")
    print("-" * 75)
    overall_stats = defaultdict(int)
    sorted_action_types = sorted(action_type_stats.keys())
    for action_type in sorted_action_types:
        stats = action_type_stats[action_type]
        for key, value in stats.items(): overall_stats[key] += value
        type_acc = (stats['type_correct'] / stats['total_steps']) * 100 if stats['total_steps'] > 0 else 0
        exact_acc = (stats['exact_correct'] / stats['total_steps']) * 100 if stats['total_steps'] > 0 else 0
        type_acc_str = f"{type_acc:.2f}% ({stats['type_correct']}/{stats['total_steps']})"
        exact_acc_str = f"{exact_acc:.2f}% ({stats['exact_correct']}/{stats['total_steps']})"
        print(f"{action_type:<20} | {type_acc_str:<25} | {exact_acc_str:<25}")
    print("-" * 75)
    total_type_acc = (overall_stats['type_correct'] / overall_stats['total_steps']) * 100 if overall_stats['total_steps'] > 0 else 0
    total_exact_acc = (overall_stats['exact_correct'] / overall_stats['total_steps']) * 100 if overall_stats['total_steps'] > 0 else 0
    total_type_acc_str = f"{total_type_acc:.2f}% ({overall_stats['type_correct']}/{overall_stats['total_steps']})"
    total_exact_acc_str = f"{total_exact_acc:.2f}% ({overall_stats['exact_correct']}/{overall_stats['total_steps']})"
    print(f"{'Overall':<20} | {total_type_acc_str:<25} | {total_exact_acc_str:<25}")
    print("="*75)

def main(summary_file: str, annotations_dir: str, episode_ids: List[str] = None):
    print(f"Loading evaluation summary: {summary_file}")    
    print(f"Loading annotation file: {annotations_dir}")
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Unable to load or parse the summary file. {e}")
        return
    if episode_ids:
        target_ids = set(episode_ids)
        summary_data = [ep for ep in summary_data if ep.get('episode_id', '').split('_')[0] in target_ids]
    
    path_results = [res for res in (calculate_path_metrics(es, annotations_dir) for es in summary_data) if res]
    
    if not path_results:
        print("\nThere are no reportable advanced indicator results.")
    else:
        grouped_by_episode = defaultdict(list)
        for res in path_results:
            grouped_by_episode[res['original_episode_id']].append(res)
        final_episode_metrics = []
        for original_id, paths in grouped_by_episode.items():
            num_paths = len(paths)
            if num_paths == 0: continue
            
            sum_total_steps_for_task = sum(p['total_steps_in_path'] for p in paths)
            sum_total_actions_for_task = sum(p['total_actions_in_path'] for p in paths)
            
            atomic_op_gp_simple = sum(p['continuous_action_count'] / p['total_actions_in_path'] if p['total_actions_in_path'] > 0 else 0 for p in paths) / num_paths
            atomic_op_count_simple = sum(p['continuous_action_count'] for p in paths) / num_paths
            atomic_op_count_weighted = sum(p['continuous_action_count'] * p['total_actions_in_path'] for p in paths) / sum_total_actions_for_task if sum_total_actions_for_task > 0 else 0
            
            step_level_gp_simple = sum(p['continuous_step_count_from_ops'] / p['total_steps_in_path'] if p['total_steps_in_path'] > 0 else 0 for p in paths) / num_paths
            
            longest_streak_simple = sum(p['max_continuous_correct_steps'] for p in paths) / num_paths
            longest_streak_weighted = sum(p['max_continuous_correct_steps'] * p['total_steps_in_path'] for p in paths) / sum_total_steps_for_task if sum_total_steps_for_task > 0 else 0
            
            initial_streak_simple = sum(p['initial_step_streak'] for p in paths) / num_paths
            initial_streak_weighted = sum(p['initial_step_streak'] * p['total_steps_in_path'] for p in paths) / sum_total_steps_for_task if sum_total_steps_for_task > 0 else 0
            initial_streak_pct_simple = sum(p['initial_step_streak'] / p['total_steps_in_path'] if p['total_steps_in_path'] > 0 else 0 for p in paths) / num_paths
            initial_streak_pct_weighted = sum(p['initial_step_streak'] for p in paths) / sum_total_steps_for_task if sum_total_steps_for_task > 0 else 0

            unique_steps_in_episode = set()
            for p in paths: unique_steps_in_episode.update(p.get('path_steps', []))
            task_length = len(unique_steps_in_episode)

            final_episode_metrics.append({
                "episode_id": original_id, "category": paths[0]['category'], "task_length": task_length,
                'atomic_op_gp_simple': atomic_op_gp_simple,
                'step_level_gp_simple': step_level_gp_simple,
                'atomic_op_count_simple': atomic_op_count_simple,
                'longest_streak_simple': longest_streak_simple,
                'initial_streak_simple': initial_streak_simple,
                'initial_streak_pct_simple': initial_streak_pct_simple,
                'atomic_op_count_weighted': atomic_op_count_weighted,
                'longest_streak_weighted': longest_streak_weighted,
                'initial_streak_weighted': initial_streak_weighted,
                'initial_streak_pct_weighted': initial_streak_pct_weighted,
                'sum_continuous_actions': sum(p['continuous_action_count'] for p in paths),
                'sum_total_actions': sum_total_actions_for_task,
                'sum_continuous_steps_from_ops': sum(p['continuous_step_count_from_ops'] for p in paths),
                'sum_total_steps': sum_total_steps_for_task,
            })
        generate_advanced_report(final_episode_metrics, "Overall Results (All Categories)")
        complex_metrics = [r for r in final_episode_metrics if 'selection' in r['category'].lower() or 'nested' in r['category'].lower()]
        if complex_metrics:
            generate_advanced_report(complex_metrics, "Results for Complex Instructions (Selection & Nested)")
    
    analyze_branching_decisions(summary_data, annotations_dir)
    analyze_by_category(summary_data, annotations_dir)
    analyze_by_action_type(summary_data, annotations_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate enhanced custom target progress metrics and report them grouped by type.")
    parser.add_argument('--summary_file', type=str, required=True, help="The path to the 'stepwise_accuracy_summary.json' file generated by the evaluation script.")
    parser.add_argument('--annotations_dir', type=str, required=True, help="The directory path containing the original JSON annotation files (e.g., '586.json').")
    parser.add_argument('--episode_ids', nargs='+', type=str, default=None, help="[Optional] Specify one or more episode IDs to be evaluated, separated by spaces.")
    args = parser.parse_args()
    main(args.summary_file, args.annotations_dir, args.episode_ids)