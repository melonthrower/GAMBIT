import json
import argparse
import os
import re
from collections import defaultdict

def analyze_data(data, annotations_dir):
    stats_template = lambda: {
        'episode_count': 0,
        'strict_success_count': 0,
        'no_complete_success_count': 0,
        'goal_progress_list': [],
        'total_steps': 0,
        'type_match_correct': 0,
        'exact_match_correct': 0,
    }
    stats_by_category = defaultdict(stats_template)

    annotations_cache = {}
    missing_annotation_files = set()

    for episode in data:
        category = episode.get('category', 'unknown')
        step_accuracy_dict = episode.get('step_accuracy', {})
        goal_progress = episode.get('goal_progress', 0.0)
        episode_id = episode.get('episode_id')
        
        if not step_accuracy_dict or not episode_id:
            continue

        if episode_id not in annotations_cache:
            annotation_file_path = os.path.join(annotations_dir, f"{episode_id}.json")
            try:
                with open(annotation_file_path, 'r', encoding='utf-8') as f:
                    annotations_cache[episode_id] = json.load(f)
            except FileNotFoundError:
                if episode_id not in missing_annotation_files:
                    print(f"Warning: Annotation file not found for episode_id '{episode_id}'.")
                    missing_annotation_files.add(episode_id)
                annotations_cache[episode_id] = None
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error parsing annotation file for episode_id '{episode_id}': {e}")
                annotations_cache[episode_id] = None

        current_annotation = annotations_cache[episode_id]
        
        is_strict_success = all(step.get('exact_match', False) for step in step_accuracy_dict.values())
        
        relevant_steps_match_status = []
        is_no_complete_success = False

        if current_annotation and 'steps' in current_annotation:
            for step_key, step_info in step_accuracy_dict.items():
                try:
                    step_index_str = step_key.split('_')[-1]
                    step_index = int(step_index_str)

                    ground_truth_step = current_annotation['steps'][step_index]
                    action = ground_truth_step.get('actions', [{}])[0]
                    action_type = action.get('action_type', 'UNKNOWN')

                    if action_type != 'complete':
                        relevant_steps_match_status.append(step_info.get('exact_match', False))
                        
                except (ValueError, IndexError, KeyError) as e:
                    print(f"Warning: Could not process step key '{step_key}' for episode '{episode_id}'. Assuming it's a relevant step. Error: {e}")
                    relevant_steps_match_status.append(step_info.get('exact_match', False))
            
            is_no_complete_success = all(relevant_steps_match_status)
        else:
            is_no_complete_success = is_strict_success
        
        stats = stats_by_category[category]
        stats['episode_count'] += 1
        stats['goal_progress_list'].append(goal_progress)
        if is_strict_success:
            stats['strict_success_count'] += 1
        if is_no_complete_success:
            stats['no_complete_success_count'] += 1

        for step_info in step_accuracy_dict.values():
            stats['total_steps'] += 1
            if step_info.get('type_match', False):
                stats['type_match_correct'] += 1
            if step_info.get('exact_match', False):
                stats['exact_match_correct'] += 1

    overall_stats = stats_template()
    for category_stats in stats_by_category.values():
        for key, value in category_stats.items():
            if isinstance(value, list):
                overall_stats[key].extend(value)
            else:
                overall_stats[key] += value
    stats_by_category['Overall'] = overall_stats
    print_table(stats_by_category)


def print_table(stats_dict):

    headers = ["Category", "Strict SR", "SR (no COMPLETE)", "Goal Progress", "Type Acc", "Exact Acc"]
    table_data = []

    desired_order = ["single", "and", "chain", "selection", "nested"]
    
    available_categories = set(k for k in stats_dict.keys() if k != 'Overall')
    
    category_keys = []
    for category in desired_order:
        if category in available_categories:
            category_keys.append(category)
            available_categories.remove(category) 
            
    if available_categories:
        category_keys.extend(sorted(list(available_categories)))

    if 'Overall' in stats_dict:
        category_keys.append('Overall')

    for category in category_keys:
        stats = stats_dict[category]
        ep_count = stats['episode_count']
        total_steps = stats['total_steps']
        strict_sr = (stats['strict_success_count'] / ep_count * 100) if ep_count > 0 else 0
        no_complete_sr = (stats['no_complete_success_count'] / ep_count * 100) if ep_count > 0 else 0
        avg_gp = (sum(stats['goal_progress_list']) / len(stats['goal_progress_list']) * 100) if stats['goal_progress_list'] else 0
        type_acc = (stats['type_match_correct'] / total_steps * 100) if total_steps > 0 else 0
        exact_acc = (stats['exact_match_correct'] / total_steps * 100) if total_steps > 0 else 0
        row = [
            category,
            f"{strict_sr:.2f}% ({stats['strict_success_count']}/{ep_count})",
            f"{no_complete_sr:.2f}% ({stats['no_complete_success_count']}/{ep_count})",
            f"{avg_gp:.2f}%",
            f"{type_acc:.2f}% ({stats['type_match_correct']}/{total_steps})",
            f"{exact_acc:.2f}% ({stats['exact_match_correct']}/{total_steps})",
        ]
        table_data.append(row)

    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, cell in enumerate(row):
            if len(cell) > col_widths[i]:
                col_widths[i] = len(cell)
    table_width = sum(col_widths) + 3 * (len(col_widths) - 1)
    print("=" * table_width)
    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    print(header_line)
    separator_line = "-|-".join('-' * width for width in col_widths)
    print(separator_line)
    for i, row in enumerate(table_data):
        if row[0] == 'Overall':
            print(separator_line)
        data_line = " | ".join(cell.ljust(col_widths[j]) for j, cell in enumerate(row))
        print(data_line)
    print("=" * table_width + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze GUI agent evaluation results and generate a formatted report.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the stepwise_accuracy_summary.json file.")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Path to the directory with original annotation JSON files (e.g., 888.json).")
    args = parser.parse_args()
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{args.input_file}' is not a valid JSON file.")
        return
    analyze_data(data, args.annotations_dir)

if __name__ == "__main__":
    main()