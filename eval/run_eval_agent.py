import os
from shutil import ExecError
import json
import random
from collections import defaultdict
from tqdm import tqdm
from utils.convert_output import convert2aitz
import argparse
from utils.evaluator import ActionEvaluator
from utils.utils import get_dataset_dir
import logging

# set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(processName)s %(message)s',
    handlers=[
        logging.FileHandler("eval_gui_agent.log"),
        logging.StreamHandler()
    ]
)

def save_stepwise_accuracy_summary(episode_results, output_path, evaluator):

    logging.info(f"Generating stepwise accuracy summary at: {output_path}")
    summary_data = []

    sorted_episode_keys = sorted(episode_results.keys())
    for episode_key in sorted_episode_keys:
        episode_steps = episode_results[episode_key]
        if not episode_steps:
            continue

        try:
            episode_id = episode_key.split('-')[-1]
        except IndexError:
            episode_id = episode_key

        category = episode_steps[0].get("category", "unknown")

        try:
            sorted_steps = sorted(episode_steps, key=lambda s: int(s.get('step_id', 0)))
        except (ValueError, TypeError):
            sorted_steps = episode_steps

        step_accuracies = {}
        for step in sorted_steps:
            step_id = step.get('step_id', 'unknown_step')
            
            eval_result = step.get("eval_result", {})

            type_match_status = eval_result.get("type_match", False)
            exact_match_status = eval_result.get("exact_match", False)

            goal_progress_value = evaluator._calculate_first_error_goal_progress(
            episode_id=episode_id,
            episode_steps=episode_steps,
            annotations_folder_path=evaluator.annobase
        )

            step_accuracies[f"step_{step_id}"] = {
                "type_match": type_match_status,
                "exact_match": exact_match_status
            }

        episode_summary = {
            "episode_id": episode_id,
            "category": category,
            "step_accuracy": step_accuracies,
            "goal_progress": goal_progress_value,
        }
        summary_data.append(episode_summary)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved stepwise accuracy summary to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save stepwise accuracy summary: {e}")
        
    
class EvalDataset(object):

    # subset of dataset to eval
    DATASET_DIR = {
        'GAMBIT': '{}/GAMBIT'
    }

    def __init__(self, data_dir, split="test", ratio=1.0) -> None:
        self.ratio = ratio
        self.data_dir = os.path.join(data_dir, split)
        self.episode_data = self._load_data_()
        self.data = self._split_to_steps_(self.episode_data)

    def _load_data_(self):

        valid_paths = defaultdict(list)
        for subset in self.DATASET_DIR:
            subdata_dir = self.DATASET_DIR[subset].format(self.data_dir)
            if os.path.exists(subdata_dir):
                sequence_names = os.listdir(subdata_dir)
                for seq_name in sequence_names:
                    seq_dir = os.path.join(subdata_dir, seq_name) 
                    
                    if not os.path.isdir(seq_dir): continue
                    episode_path = os.path.join(seq_dir, f"{seq_name}.json")
                    valid_paths[subset].append(episode_path)

        sampled_paths = []
        for subset, v_paths in valid_paths.items():
            N = len(v_paths)
            k = int(self.ratio * N)
            sampled_paths += random.sample(v_paths, k) if self.ratio < 1.0 else v_paths

        ep_data = []
        for episode_path in sampled_paths:
            try:
                with open(episode_path, "r") as f:
                    episode_data = json.load(f)
                    ep_data.append(episode_data)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding failed, file: {episode_path}, error: {e}")
            except Exception as e:
                logging.error(f"Error occurred when loading file {episode_path}: {e}")
        return ep_data

    def _split_to_steps_(self, episode_data):

        data = []
        for edx, episode in enumerate(episode_data):
            for idx, step in enumerate(episode):
                try:
                    step['image_full_path'] = os.path.join(self.data_dir, step['image_path'])
                    data.append(step)
                except KeyError as e:
                    logging.error(f"Missing key {e}, at episode {edx}, step {idx}")
                except Exception as e:
                    logging.error(f"Error processing episode {edx}, step {idx}: {e}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def process_step_data(step_data, evaluator, save_dir):

    subset = step_data.get('subset')
    episode_id = step_data.get('episode_id')
    step_id = step_data.get('step_id')

    if subset is None or episode_id is None or step_id is None:
        raise ValueError(f"Missing subset/episode_id/step_id in test step data: {step_data}")

    save_dir_ep = os.path.join(save_dir, f"{subset}-{episode_id}") 
    cur_save_path = os.path.join(save_dir_ep, f"{subset}-{episode_id}_{step_id}.json")

    try:
        os.makedirs(os.path.dirname(cur_save_path), exist_ok=True)

        if not os.path.exists(cur_save_path):
            return None

        # Load the existing file
        with open(cur_save_path, "r") as file:
            try:
                pred = json.load(file)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding failed, file: {cur_save_path}, error: {e}")
                pred = {'action_predict': {'COA': {'txt': {'ACTION': None, 'ARGS': None, 'STATUS': None}}}}

        assert pred is not None

        # Use global evaluator to evaluate
        result = evaluator(step_data, pred)
        
        # Nest the evaluation result into the step_data. This is crucial for metric calculation.
        step_data['eval_result'] = result
        
        return step_data

    except Exception as e:
        raise  ExecError(f"An error occurred, indicating unhandled edge case.{cur_save_path}")

        
def evaluate(args):

    # Get dataset path

    args.data_dir, args.split, _ = get_dataset_dir(args.data_name)
    # Convert to aitz format
    convert2aitz(os.path.abspath(args.input_path), os.path.abspath(args.output_dir), max_workers=16)

    save_dir = os.path.abspath(args.output_dir)
    results_save_file = os.path.join(save_dir, "result.json")

    # Initialize dataset
    eval_data = EvalDataset(data_dir=args.data_dir) 
    logging.info(f"Total steps: {len(eval_data)}, total episodes: {len(eval_data.episode_data)}.")

    evaluator = ActionEvaluator(save_dir, args.eval_android_control)
    results = list(tqdm(map(process_step_data, eval_data.data, [evaluator]*len(eval_data.data), [save_dir]*len(eval_data.data)),total=len(eval_data.data), desc="Processing steps", ncols=100))
    results = list(filter(lambda x: x is not None, results))

    # Save results
    try:
        os.makedirs(os.path.dirname(results_save_file), exist_ok=True)
        with open(results_save_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"Evaluation results saved to {results_save_file}")
    except Exception as e:
        logging.error(f"Error saving results to {results_save_file}: {e}")

    # Aggregate episode results
    episode_results = defaultdict(list)
    nested_list = set()
    for result in results:
        subset = result.get("subset")
        episode_id = result.get("episode_id")
        if subset is None or episode_id is None:
            logging.warning(f"Result missing subset/episode_id: {result}")
            continue
        episode_key = f"{subset}-{episode_id}"
        episode_results[episode_key].append(result)
        if(result['category'] == 'nested') :
            nested_list.add(result["episode_id"])

    # Compute final evaluation metrics
    stepwise_summary_path = os.path.join(args.output_dir, 'stepwise_accuracy_summary.json')
    save_stepwise_accuracy_summary(episode_results, stepwise_summary_path, evaluator)
    try:
        episode_metrics = evaluator.compute_episode_metrics(episode_results)
        results_eval_only = [r['eval_result'] for r in results if 'eval_result' in r]
        atomic_metrics = ActionEvaluator.compute_atomic_metrics(results_eval_only)
        atomic_instruction_metrics = ActionEvaluator.compute_atomic_instruction_metrics(results)

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logging.info(f"episode_metrics: {episode_metrics}")
        logging.info(f"atomic_metrics: {atomic_metrics}")
        logging.info(f"atomic_instruction_metrics: {atomic_instruction_metrics}") 

        logging.info(
            f"success_rate: {episode_metrics.get('success_rate')}, "
            f"goal_progress: {episode_metrics.get('goal_progress')}, "
            f"type_acc: {atomic_metrics.get('total', {}).get('type_acc')}, "
            f"exact_acc: {atomic_metrics.get('total', {}).get('exact_acc')}"
        )

        summary_save_file = os.path.join(args.output_dir, 'summary.json')
        logging.info(f"Evaluation summary saved to {summary_save_file}")
        with open(summary_save_file, 'w') as f:
            json.dump(episode_metrics|atomic_metrics, f, ensure_ascii=False)

    except Exception as e:
        logging.error(f"Error computing evaluation metrics: {e}")

# ==========================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GUI Agent Eval")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input prediction JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--data_name", type=str, default="GAMBIT", help="Eval dataset name")
    parser.add_argument("--eval_android_control", action="store_true", help="For evaluating android control, which is different from other datasets according to qwen's scripts")
    args = parser.parse_args()

    logging.info(f"Received arguments: {args}")
    random.seed(args.seed)

    evaluate(args)
