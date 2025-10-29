import base64
import sys
import argparse
import re
import copy
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_DEPRECATED_BEAM_SEARCH"]="1"
import json
import yaml
import time
import torch
import random
from yacs.config import CfgNode as CN
import re
import numpy as np
import requests
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.qwen_mobile_tool import aitw_2_uitars
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir

import traceback

DEVICES = [
    "cuda:0",
    "cuda:1",
    "cuda:2","cuda:3"
    ]
torch.set_num_threads(4)

RAW_OUTPUT_DIR = None 

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

ERROR_LOG_DIR = os.path.join(current_dir, "ui_tars_error")
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

if current_dir not in sys.path:
    sys.path.append(current_dir)
    
def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)

NO_THOUGHT_EXAMPLE = {"Press":"BACK"}
SYSTEM_PROMPT = "You are a helpful assistant."

_llm = None
_tokenizer = None

def transform_history_string_to_dict(action_list: list) -> dict:
    """
    Transforms a history action from string format (from t.py) to the dict format
    expected by aitw_2_uitars.
    """
    if not action_list or not isinstance(action_list, list) or not action_list[0]:
        return {}

    action_str = action_list[0]
    action_dict = {}

    if action_str.startswith("TYPE"):
        text = action_str.split(":", 1)[1].strip()
        action_dict['result_action_type'] = 'type'
        action_dict['result_action_text'] = text
        
    elif action_str.startswith("CLICK") or action_str.startswith("LONG_PRESS"):
        action_type = "long_press" if action_str.startswith("LONG_PRESS") else "click"
        coords_str = action_str.split(":", 1)[1].strip(" ()")
        x1, y1, x2, y2 = map(float, coords_str.split(","))
        touch_x = ((x1 + x2) / 2) / 1000.0
        touch_y = ((y1 + y2) / 2) / 1000.0
        action_dict['result_action_type'] = action_type
        action_dict['result_touch_yx'] = json.dumps([touch_y, touch_x])
        action_dict['result_lift_yx'] = json.dumps([touch_y, touch_x])

    elif action_str.startswith("SCROLL"):
        action_dict['result_action_type'] = 'scroll'
        direction = action_str.split(":", 1)[1].strip().lower()
        if direction == 'up':
            touch_yx = [0.7, 0.5]; lift_yx = [0.3, 0.5]
        elif direction == 'down':
            touch_yx = [0.3, 0.5]; lift_yx = [0.7, 0.5]
        elif direction == 'left':
            touch_yx = [0.5, 0.7]; lift_yx = [0.5, 0.3]
        else: # right
            touch_yx = [0.5, 0.3]; lift_yx = [0.5, 0.7]
        action_dict['result_touch_yx'] = json.dumps(touch_yx)
        action_dict['result_lift_yx'] = json.dumps(lift_yx)
            
    elif action_str.startswith("PRESS_BACK"):
        action_dict['result_action_type'] = 'press_back'
    
    elif action_str.startswith("PRESS_HOME"):
        action_dict['result_action_type'] = 'press_home'

    elif action_str.startswith("COMPLETE"):
        action_dict['result_action_type'] = 'complete'
        
    elif action_str.startswith("IMPOSSIBLE"):
        action_dict['result_action_type'] = 'impossible'
        
    elif action_str.startswith("WAIT"):
        action_dict['result_action_type'] = 'wait'

    return action_dict

def save_raw_output(raw_output_dir, episode, raw_text):
    if not raw_output_dir:
        return
    try:
        category = episode.get("category", "unknown_category")
        episode_id = episode.get("episode_id", "unknown_episode")
        step_id = episode.get("step_id", "unknown_step")
        
        log_subdir = os.path.join(raw_output_dir, category)
        os.makedirs(log_subdir, exist_ok=True)

        filename = f"{episode_id}_step_{step_id}.txt"
        filepath = os.path.join(log_subdir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(raw_text)
            
    except Exception as e:
        print(f"Error saving raw output: {e}")
        
def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,   trust_remote_code=True,  torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    if _tokenizer is None:
        _tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

def move_to(device):
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"

def run_episode_low(episode, image_path, history_list, use_low_instruction, raw_output_dir=None):

    global _llm, _tokenizer
    torch.cuda.empty_cache()

    history_messages = []
    history_screenshots = episode.get("history_screenshot", [])
    history_actions = episode.get("history_action", [])
    history_instructs = episode.get("history_instruct", [])

    num_hist_steps = len(history_screenshots)
    start_index = max(0, num_hist_steps - 4)

    for i in range(start_index, num_hist_steps):
        hist_image_path = history_screenshots[i]
        if os.path.exists(hist_image_path):
            history_messages.append({
                "role": "user",
                "content": [{"type": "image", "image": hist_image_path}]
            })
        else:
            print(f"files not find: {hist_image_path}")
            continue
        
        action_dict = transform_history_string_to_dict(history_actions[i])
        action_str = aitw_2_uitars(action_dict)

        thought_str = history_instructs[i]
        history_messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"Thought: {thought_str}\nAction: {action_str}"}]
        })

    instruction = episode["instruction"]
    low_instruction = episode["low_instruction"]
    thought = "Thought: " + low_instruction + "\nAction:"
    
    text = ("You are a GUI agent. You are given a task and your action history, with screenshots. "
            "You need to perform the next action to complete the task. \n\n"
            "## Output Format\n\n"
            "Thought: ...\n"
            "Action: ...\n\n\n"
            "## Action Space\n"
            "click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')\n"
            "long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')\n"
            "type(content=\'\')\n"
            "scroll(direction=\'down or up or right or left\')\n"
            "press_back()\n"
            "press_home()\n"
            "wait()\n"
            "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
            "## Note\n"
            "- Use English in Thought part.\n\n"
            "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
            "## User Instruction\n" + instruction)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    conversation.extend(history_messages)
    conversation.append({
        "role": "user",
        "content": [{"type": "image", "image": image_path}],
    })
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": thought}],
    })

    text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    text_prompt = text_prompt.rsplit("<|im_end|>", 1)[0].strip()
    # print(text_prompt)
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = _tokenizer(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = _llm.device
    inputs = inputs.to(device)

    output_ids = _llm.generate(**inputs, max_new_tokens=128, temperature=0.1)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = _tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    raw_model_output = output_text[0]
    if raw_output_dir:
        save_raw_output(raw_output_dir, episode, raw_model_output)

    episode["pred"] = uitars2minicpm(output_text[0])
    return episode

def run_episode_high(episode, image_path, history_list, use_low_instruction, raw_output_dir):

    global _llm, _tokenizer
    torch.cuda.empty_cache()

    history_messages = []
    history_screenshots = episode.get("history_screenshot", [])
    history_actions = episode.get("history_action", [])
    history_instructs = episode.get("history_instruct", [])
    
    num_hist_steps = len(history_screenshots)
    start_index = max(0, num_hist_steps - 4)

    for i in range(start_index, num_hist_steps):
        hist_image_path = history_screenshots[i]
        if os.path.exists(hist_image_path):
            history_messages.append({
                "role": "user",
                "content": [{"type": "image", "image": hist_image_path}]
            })
        else:
            print(f"files not found: {hist_image_path}")
            continue
        
        action_dict = transform_history_string_to_dict(history_actions[i])
        action_str = aitw_2_uitars(action_dict)

        thought_str = history_instructs[i]
        history_messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"Thought: {thought_str}\nAction: {action_str}"}]
        })

    instruction = episode["instruction"]
    text = ("You are a GUI agent. You are given a task and your action history, with screenshots. "
            "You need to perform the next action to complete the task. \n\n"
            "## Output Format\n\n"
            "Thought: ...\n"
            "Action: ...\n\n\n"
            "## Action Space\n"
            "click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')\n"
            "long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')\n"
            "type(content=\'\')\n"
            "scroll(direction=\'down or up or right or left\')\n"
            "press_back()\n"
            "press_home()\n"
            "wait()\n"
            "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
            "## Note\n"
            "- Use English in Thought part.\n\n"
            "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
            "## User Instruction\n" + instruction)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    conversation.extend(history_messages)
    conversation.append({
        "role": "user",
        "content": [{"type": "image", "image": image_path}],
    })

    text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # print(text_prompt) # Commented out for cleaner logs
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = _tokenizer(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = _llm.device
    inputs = inputs.to(device)
    # print(text_prompt)
    output_ids = _llm.generate(**inputs, max_new_tokens=128, temperature=0.1)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = _tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    
    raw_model_output = output_text[0]
    save_raw_output(raw_output_dir, episode, raw_model_output)
    
    episode["pred"] = uitars2minicpm(output_text[0])
    return episode


def uitars2minicpm(action_str):
    result = {"STATUS": "continue"}
    
    def extract_coords(s):
        first_bracket = s.find("(")
        start = s.find("(", first_bracket + 1)
        end = s.find(")")
        if start != -1 and end != -1:
            coords_str = s[start+1:end].strip()
            x, y = coords_str.split(",")
            return [int(x), int(y)]
        raise ValueError(f"Cannot find coordinates in the string: {s}")
    
    if "click(" in action_str:
        try: result["POINT"] = extract_coords(action_str)
        except: pass
        
    elif "long_press(" in action_str:
        try: result["POINT"] = extract_coords(action_str)
        except: pass
        if "time='" in action_str:
            time = action_str.split("time='")[1].split("'")[0]
            result["duration"] = int(time) if time else 1000
            
    elif "type(" in action_str:
        try:
            content = action_str.split("content='")[1].split("'")[0]
            result["TYPE"] = content
        except: pass
        
    elif "scroll(" in action_str:
        try:
            direction = action_str.split("direction='")[1].split("'")[0]
            result["POINT"] = [500, 500]
            if direction == "down": direction = "up"
            elif direction == "up": direction = "down"
            elif direction == "right": direction = "left"
            elif direction == "left": direction = "right"
            result["to"] = direction
        except: pass

    elif "press_back()" in action_str:
        result["PRESS"] = "BACK"
        
    elif "press_home()" in action_str:
        result["PRESS"] = "HOME"
        
    elif "wait()" in action_str:
        result["duration"] = 200
        
    elif "finished()" in action_str:
        result["STATUS"] = "finish"
        
    elif "open_app(app_name=" in action_str:
        try: result["OPEN_APP"] = action_str.split("app_name='")[1].split("'")[0]
        except: pass
    else:
        print(f"Warning, invalid action format detected: {action_str}")
        
    return result

def run_episode(episode, image_path, history_list, use_low_instruction, raw_output_dir):
    if use_low_instruction:
        return run_episode_low(episode, image_path, [], use_low_instruction, raw_output_dir)
    else:
        return run_episode_high(episode, image_path, [], use_low_instruction, raw_output_dir)

def load_image(episode, image_path, history_list, use_low_instruction):
    return (episode, image_path, [], use_low_instruction)

def predict(args, datasets):
    global USE_LOW_INSTRUCTION, RAW_OUTPUT_DIR
    USE_LOW_INSTRUCTION = args.use_low_instruction
    
    RAW_OUTPUT_DIR = os.path.join(args.output_dir, "raw_model_outputs")
    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
    
    data_dir = args.data_dir
    split_type = args.split
    print("Predicting on:", datasets)
    
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    processed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=len(DEVICES), initializer=_init_llm, initargs=(args.model_path,)) as poolexec:
        tasks = []
        print("Moving model to devices")
        for device in DEVICES:
            tasks.append(poolexec.submit(move_to, device))
        for t in tasks:
            print(t.result())
    
        for dataset in datasets:
            save_dir = os.path.join(args.output_dir, dataset)
            os.makedirs(save_dir, exist_ok=True)
                
            episode_dir = os.path.join(data_dir, split_type, dataset)
            output_file = os.path.join(save_dir, "predict.jsonl")
            
            if not os.path.exists(episode_dir):
                print(f"Warning: Episode directory not found, skipping: {episode_dir}")
                continue
            
            episodes_files = os.listdir(episode_dir)
            
            all_tasks_to_submit = []
            print("Loading and verifying episodes...")
            with ThreadPoolExecutor(max_workers=16) as executor:
                loading_futures = []
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
                            image_path = episode.get("image_path", "")

                        if not image_path or not os.path.exists(image_path):
                            error_id = f"{dataset}_{episode.get('episode_id', 'N/A')}_{episode.get('step_id', 'N/A')}"
                            error_message = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, FileNotFoundError: {image_path}, id: {error_id}\n"
                            with open(os.path.join(ERROR_LOG_DIR, 'missing_files.log'), 'a', encoding='utf-8') as f_log:
                                f_log.write(error_message)
                            skipped_count += 1
                            continue

                        episode_copy = copy.deepcopy(episode)
                        loading_futures.append(executor.submit(load_image, episode_copy, image_path, [], USE_LOW_INSTRUCTION))

                for f in as_completed(loading_futures):
                    try:
                        all_tasks_to_submit.append(f.result())
                    except Exception as exc:
                        print(f'An exception occurred during task loading: {exc}')
                        skipped_count += 1

            with open(output_file, "w", encoding="utf-8") as f_out:
                print(f"Predicting {len(all_tasks_to_submit)} episodes for {dataset}...")
                
                future_to_info = {}
                for task_value in all_tasks_to_submit:
                    episode_data = task_value[0]
                    future = poolexec.submit(run_episode, *task_value, RAW_OUTPUT_DIR)
                    info = {
                        "category": episode_data.get("category", "N/A"),
                        "episode_id": episode_data.get("episode_id", "N/A"),
                        "step_id": episode_data.get("step_id", "N/A")
                    }
                    future_to_info[future] = info

                for task in tqdm(as_completed(future_to_info), total=len(future_to_info), dynamic_ncols=True):
                    episode_info = future_to_info[task]
                    try:
                        episode = task.result()
                        episode_json = json.dumps(episode, ensure_ascii=False)
                        f_out.write(episode_json + "\n")
                        f_out.flush()
                        processed_count += 1
                    except Exception as e:
                        skipped_count += 1
                        error_id = f"{episode_info['category']}_{episode_info['episode_id']}_{episode_info['step_id']}"
                        error_message = (f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, "
                                       f"Error during prediction task for id: {error_id}\n"
                                       f"Error Type: {type(e).__name__}, Message: {e}\n"
                                       f"{traceback.format_exc()}\n"
                                       "--------------------------------\n")
                        with open(os.path.join(ERROR_LOG_DIR, 'prediction_errors.log'), 'a', encoding='utf-8') as f_log:
                            f_log.write(error_message)
                        continue

        print(f"Prediction saved at: {args.output_dir}.")
    os.system(f"cat {args.output_dir}/*/predict.jsonl > {args.output_dir}/all.jsonl")
    print(f"Merged prediction saved at: {args.output_dir}/all.jsonl.")
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Successfully processed episodes: {processed_count}")
    print(f"Skipped episodes due to errors: {skipped_count}")
    print(f"Total episodes attempted: {processed_count + skipped_count}")
    print(f"Check '{ERROR_LOG_DIR}' directory for details on skipped files.")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UI-TARS Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/home/test/test03/models/UI-TARS-7B-SFT"),
                       help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.getenv('OUTPUT_PATH', "eval_results")),
                       help="Directory to save results")
    parser.add_argument("--data_name", type=str, default=os.getenv("PREDICT_DATASET", "GAMBIT"),
                       help="Eval dataset name")
    parser.add_argument("--use_low_instruction", action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)

    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    
    print(f'Loading model at : {args.model_path}')
    print(f'Loading data at  : {args.data_dir}')
    print(f'Processing subsets: {data_subset}')
    print(f'Saving results at: {args.output_dir}')
    
    predict(args, data_subset)