import sys
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import random
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir
import argparse
import logging
import time
from pprint import pprint

DEVICES = ["cuda:0","cuda:1",
          "cuda:2", "cuda:3"
          ]
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if current_dir not in sys.path:
    sys.path.append(current_dir)

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)

ACTION_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema.json'), encoding="utf-8"))
items = list(ACTION_SCHEMA.items())

ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = f'''# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema

{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''

EXTRACT_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))


_llm = None
_tokenizer = None

def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.bfloat16)
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def move_to(device):
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"


def run_episode(episode, msg,):
    global _llm,_tokenizer
    outputs = _llm.chat(image=None, msgs=msg, system_prompt=SYSTEM_PROMPT, tokenizer=_tokenizer, temperature=0.1,top_p=0.3,n=1,)
    episode["pred"] = extract_and_validate_json(outputs)
    return episode


def extract_and_validate_json(input_string):
    try:
        json_obj = json.loads(input_string)
        jsonschema.validate(json_obj, EXTRACT_SCHEMA)
        return json_obj
    except json.JSONDecodeError as e:
        print("Error, JSON is NOT valid.")
        return input_string
    except Exception as e:
        print(f"Error, JSON is NOT valid according to the schema.{input_string}", e)
        return input_string

def load_image(episode, image_path, data_name):
    """
    
    """
    def __resize__(origin_img):
        resolution = origin_img.size
        w, h = resolution
        max_line_res = 1120
        if max_line_res is not None:
            max_line = max_line_res
            if h > max_line:
                w = int(w * max_line / h)
                h = max_line
            if w > max_line:
                h = int(h * max_line / w)
                w = max_line
        img = origin_img.resize((w, h), resample=Image.Resampling.LANCZOS)
        return img

    try:
        current_image = Image.open(image_path).convert("RGB")
        current_image = __resize__(current_image)
    except FileNotFoundError:
        error_dir = os.path.join(current_dir, "minicpm_error")
        os.makedirs(error_dir, exist_ok=True)
        error_log_file = os.path.join(error_dir, "missing_files.log")

        error_id = f"{episode.get('episode_id', 'N/A')}_{episode.get('step_id', 'N/A')}"
        error_message = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, FileNotFoundError: {image_path}, id: {error_id}\n"
        
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(error_message)
            
        print(f"pictures not found {image_path} (id: {error_id})")
        return None
 
    if args.use_low_instruction:
        query = episode['low_instruction']
    else:
        query = episode['instruction']

    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                f"<Question>{query}</Question>\n当前屏幕截图：",
                current_image
            ]
        }
    )
    # print(messages)
    return (episode, messages)


def predict(args):
    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name) 
    print(f"Predicting on: {args.data_dir}/{args.split}")
    print(f"Data subset: {data_subset}")
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    batch_size = 100
    episode_infos = []

    for dataset in data_subset:
        save_dir = os.path.join(args.output_dir, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        episode_dir = os.path.join(args.data_dir, args.split, dataset)  
        output_file = os.path.join(save_dir, "predict.jsonl")

        if not os.path.exists(episode_dir):
            continue

        episodes_files = os.listdir(episode_dir)
        for episodes_file in episodes_files:
            episodes_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}.json")
            try:
                with open(episodes_path, 'r', encoding='utf-8') as f:
                    episodes = json.load(f)
            except Exception as e:
                print(f"Failed to load {episodes_path}: {e}")
                continue

            for episode in episodes:
                episode["category"] = dataset
                image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{episode['step_id']}.jpeg")
                if not os.path.exists(image_path):
                    image_path = image_path.replace(".jpeg", ".png")
                    if not os.path.exists(image_path):
                        image_path = episode['image_path']
                episode_infos.append((episode, image_path, args.data_name, output_file))

    print(f"Total episode count: {len(episode_infos)}")

    from math import ceil
    from collections import defaultdict

    output_batches = defaultdict(list)
    for info in episode_infos:
        output_batches[info[3]].append(info)

    processed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=len(DEVICES), initializer=_init_llm, initargs=(args.model_path,)) as poolexec:
        print("Moving model to devices")
        futures = [poolexec.submit(move_to, dev) for dev in DEVICES]
        for fut in futures:
            print(fut.result())

        for output_file, episode_batch in output_batches.items():
            total_batches = ceil(len(episode_batch) / batch_size)
            print(f"Processing {len(episode_batch)} episodes for output: {output_file} in {total_batches} batches.")
            with open(output_file, "w", encoding="utf-8") as f_out:
                for batch_idx in range(total_batches):
                    start = batch_idx * batch_size
                    end = min((batch_idx + 1) * batch_size, len(episode_batch))
                    batch = episode_batch[start:end]

                    load_futures = []
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        for episode, image_path, data_name, _ in batch:
                            load_futures.append(executor.submit(load_image, episode, image_path, data_name))
                        batch_results = []
                        for f in as_completed(load_futures):
                            try:
                                result = f.result()
                                if result is not None:
                                    batch_results.append(result)
                                else:

                                    skipped_count += 1

                            except Exception as e:
                                print(f"Error retrieving result from future: {e}")
                                skipped_count += 1
                    
                    future_to_info = {}
                    for episode_data, messages in batch_results:
                        future = poolexec.submit(run_episode, episode_data, messages)

                        info = {
                            "episode_id": episode_data.get("episode_id", "UNKNOWN_ID"),
                            "step_id": episode_data.get("step_id", "UNKNOWN_STEP")
                        }
                        future_to_info[future] = info

                    for task in tqdm(as_completed(future_to_info), total=len(future_to_info), dynamic_ncols=True, desc=f"Predicting batch {batch_idx+1}/{total_batches}"):

                        episode_info = future_to_info[task]
                        try:
                            episode = task.result()
                            episode_json = json.dumps(episode, ensure_ascii=False)
                            f_out.write(episode_json + "\n")
                            f_out.flush()
                            processed_count += 1
                        except Exception as e:
                            skipped_count += 1

                            error_id = f"{episode_info['episode_id']}_{episode_info['step_id']}"
                            error_message = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, Error during prediction task: {e}, id: {error_id}"
                            
                            print(f"\nError: {e}, id: {error_id}")
                            
                            error_dir = os.path.join(current_dir, "minicpm_error")
                            os.makedirs(error_dir, exist_ok=True)
                            log_file = os.path.join(error_dir, "prediction_errors.log")
                            with open(log_file, 'a', encoding='utf-8') as f_log:
                                f_log.write(error_message + "\n")
                            
                            continue
            print(f"Prediction saved at: {output_file}.")

    os.system(f"cat {args.output_dir}/*/predict.jsonl > {args.output_dir}/all.jsonl")
    print(f"Merged prediction saved at: {args.output_dir}/all.jsonl.")

    return processed_count, skipped_count

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GUI Agent Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--data_name", type=str, default="GAMBIT", help="Eval dataset name")
    parser.add_argument("--use_low_instruction", action='store_true', help="If set, use low-level instructions for the model.")
    args = parser.parse_args()
    random.seed(args.seed)

    print(f'Loading model at : {args.model_path}')
    print(f'Saving results at: {args.output_dir}')

    processed_count, skipped_count = predict(args)

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)

    print(f"Successfully processed episodes: {processed_count}")
    print(f"Skipped episodes due to errors: {skipped_count}")
    print(f"Final evaluation metrics will be based on {processed_count} samples.")
    print("Check 'minicpm_error/' directory for details on skipped files.")
    print("="*50 + "\n")
