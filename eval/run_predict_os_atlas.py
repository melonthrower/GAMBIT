# --- START OF FILE run_predict_os_atlas.py (MODIFIED) ---

import base64
import sys
import argparse
import json
import re
import copy
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_DEPRECATED_BEAM_SEARCH"]="1"
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
# from utils.qwen_mobile_tool import aitw_2_uitars # This import seems unused
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir

DEVICES = [
    "cuda:0", 
    "cuda:1", 
    "cuda:2", "cuda:3",
    #   "cuda:4", "cuda:5", "cuda:6", "cuda:7",
    ]
torch.set_num_threads(4)
# USE_LOW_INSTRUCTION = False

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if current_dir not in sys.path:
    sys.path.append(current_dir)
    
def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)

NO_THOUGHT_EXAMPLE = {"Press":"BACK"}
SYSTEM_PROMPT = "You are a helpful assistant."

_llm = None
_tokenizer = None

def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,   trust_remote_code=True,  torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    if _tokenizer is None:
        _tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

def move_to(device):
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"

def build_history_actions_str(history_instruct_list):
    """
    Builds a string from the list of historical low-level instructions.
    The input `history_instruct_list` comes directly from the preprocessed data.
    """
    if not history_instruct_list:
        return "None"
    history = ""
    for i, instruction in enumerate(history_instruct_list):
        history += f"Step {i+1}: {instruction}\n"
    return history

def run_episode_high(episode, image_path, history_list, use_low_instruction):
    try:
        global _llm, _tokenizer
        torch.cuda.empty_cache()
        
        instruction = episode["instruction"]
        # history_instruct_list = history_list.get("history_instruct", [])
        # history_str = build_history_actions_str(history_instruct_list)
        
        text = f"""\nYou are a foundational action model capable of automating tasks across various digital environments, including desktop systems like Windows, macOS, and Linux, as well as mobile platforms such as Android and iOS. You also excel in web browser environments. You will interact with digital devices in a human-like manner: by reading screenshots, analyzing them, and taking appropriate actions.\n\nYour expertise covers two types of digital tasks:\n    - Grounding: Given a screenshot and a description, you assist users in locating elements mentioned. Sometimes, you must infer which elements best fit the description when they aren't explicitly stated.\n    - Executable Language Grounding: With a screenshot and task instruction, your goal is to determine the executable actions needed to complete the task.\n\n\nYou are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:\n\n1. Basic Actions\nBasic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. \nBasic Action 1: CLICK \n    - purpose: Click at the specified position.\n    - format: CLICK <point>[[x-axis, y-axis]]</point>\n    - example usage: CLICK <point>[[101, 872]]</point>\n       \nBasic Action 2: TYPE\n    - purpose: Enter specified text at the designated location.\n    - format: TYPE [input text]\n    - example usage: TYPE [Shanghai shopping mall]\n\nBasic Action 3: SCROLL\n    - purpose: Scroll in the specified direction.\n    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n    - example usage: SCROLL [UP]\n    \n2.Custom Actions\nCustom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n\n\nCustom Action 1: LONG_PRESS \n    - purpose: Long press at the specified position.\n    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>\n    - example usage: LONG_PRESS <point>[[101, 872]]</point>\n\nCustom Action 2: PRESS_BACK\n    - purpose: Press a back button to navigate to the previous screen.\n    - format: PRESS_BACK\n    - example usage: PRESS_BACK\n\nCustom Action 3: PRESS_HOME\n    - purpose: Press a home button to navigate to the home page.\n    - format: PRESS_HOME\n    - example usage: PRESS_HOME\n\nCustom Action 4: PRESS_RECENT\n    - purpose: Press the recent button to view or switch between recently used applications.\n    - format: PRESS_RECENT\n    - example usage: PRESS_RECENT\n\nCustom Action 5: IMPOSSIBLE\n    - purpose: Wait for the screen to load.\n    - format: WAIT\n    - example usage: WAIT\n\nCustom Action 6: COMPLETE\n    - purpose: Indicate the task is finished.\n    - format: COMPLETE\n    - example usage: COMPLETE\n\n\nIn most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.\nThoughts: Clearly outline your reasoning process for current step.\nActions: Specify the actual actions you will take based on your reasoning.\n\nYour current task instruction, action history, and associated screenshot are as follows:\nScreenshot: """
        text2 = f"""\nTask: {instruction}\nHistory: \nNone\n"""
        
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image", "image": image_path}, {"type": "text", "text": text2}]},
        ]

        text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = _tokenizer(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        # print('-'*30) # Optional: uncomment for debugging
        # print(f"history_str:{history_str}")
        # print(episode["episode_id"])
        # print(episode["step_id"])
        device = _llm.device
        inputs = inputs.to(device)
        
        output_ids = _llm.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        episode["pred"] = os_atlas_2minicpm(output_text[0], use_low_instruction)
    except Exception as e:
        import traceback
        print(f"Error in run_episode_high: {e}")
        traceback.print_exc()
        # Re-raise the exception so it can be caught by the main prediction loop
        raise e
    return episode

def run_episode_low(episode, image_path, history_list, use_low_instruction):
    try:
        global _llm, _tokenizer
        torch.cuda.empty_cache()
        
        instruction = episode["instruction"]
        low_instruction = episode["low_instruction"]
        
        history_instruct_list = history_list.get("history_instruct", [])
        history_str = build_history_actions_str(history_instruct_list)
        
        text = f"""\nYou are a foundational action model capable of automating tasks across various digital environments, including desktop systems like Windows, macOS, and Linux, as well as mobile platforms such as Android and iOS. You also excel in web browser environments. You will interact with digital devices in a human-like manner: by reading screenshots, analyzing them, and taking appropriate actions.\n\nYour expertise covers two types of digital tasks:\n    - Grounding: Given a screenshot and a description, you assist users in locating elements mentioned. Sometimes, you must infer which elements best fit the description when they aren't explicitly stated.\n    - Executable Language Grounding: With a screenshot and task instruction, your goal is to determine the executable actions needed to complete the task.\n\n\nYou are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:\n\n1. Basic Actions\nBasic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. \nBasic Action 1: CLICK \n    - purpose: Click at the specified position.\n    - format: CLICK <point>[[x-axis, y-axis]]</point>\n    - example usage: CLICK <point>[[101, 872]]</point>\n       \nBasic Action 2: TYPE\n    - purpose: Enter specified text at the designated location.\n    - format: TYPE [input text]\n    - example usage: TYPE [Shanghai shopping mall]\n\nBasic Action 3: SCROLL\n    - purpose: Scroll in the specified direction.\n    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n    - example usage: SCROLL [UP]\n    \n2.Custom Actions\nCustom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n\n\nCustom Action 1: LONG_PRESS \n    - purpose: Long press at the specified position.\n    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>\n    - example usage: LONG_PRESS <point>[[101, 872]]</point>\n\nCustom Action 2: PRESS_BACK\n    - purpose: Press a back button to navigate to the previous screen.\n    - format: PRESS_BACK\n    - example usage: PRESS_BACK\n\nCustom Action 3: PRESS_HOME\n    - purpose: Press a home button to navigate to the home page.\n    - format: PRESS_HOME\n    - example usage: PRESS_HOME\n\nCustom Action 4: PRESS_RECENT\n    - purpose: Press the recent button to view or switch between recently used applications.\n    - format: PRESS_RECENT\n    - example usage: PRESS_RECENT\n\nCustom Action 5: IMPOSSIBLE\n    - purpose: Wait for the screen to load.\n    - format: WAIT\n    - example usage: WAIT\n\nCustom Action 6: COMPLETE\n    - purpose: Indicate the task is finished.\n    - format: COMPLETE\n    - example usage: COMPLETE\n\n\nIn most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.\nThoughts: Clearly outline your reasoning process for current step.\nActions: Specify the actual actions you will take based on your reasoning.\n\nYour current task instruction, action history, and associated screenshot are as follows:\nScreenshot: """
        text2 = f"""\nTask: {instruction} You need to: {low_instruction}\nHistory: \n{history_str}\n"""
        
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image", "image": image_path}, {"type": "text", "text": text2}]},
        ]

        text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # print('-'*30) # Optional: uncomment for debugging
        # print(f"history_str:{history_str}")
        
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = _tokenizer(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        device = _llm.device
        inputs = inputs.to(device)

        output_ids = _llm.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        episode["pred"] = os_atlas_2minicpm(output_text[0], use_low_instruction)
    except Exception as e:
        import traceback
        print(f"Error in run_episode_low: {e}")
        traceback.print_exc()
        
        raise e
    return episode

import re

def os_atlas_2minicpm(action_str, use_low_instruction):
    result = {"STATUS": "continue"}
    try:
        action_start = -1
        if "Actions:" in action_str:
             action_start = action_str.find("Actions:")
             action_content = action_str[action_start + len("Actions:"):].strip()
        elif "actions:" in action_str:
            action_start = action_str.find("actions:")
            action_content = action_str[action_start + len("actions:"):].strip()
        else: 
            action_content = action_str.strip()

        if "CLICK" in action_content:
            coords_match = re.search(r'\[\[\s*(\d+)\s*,\s*(\d+)\s*\]\]', action_content)
            if coords_match:
                x, y = map(int, coords_match.groups())
                result["POINT"] = [x, y]
        elif "TYPE" in action_content:
            text_match = re.search(r'\[(.*?)\]', action_content)
            if text_match:
                result["TYPE"] = text_match.group(1)
        elif "SCROLL" in action_content:
            direction_match = re.search(r'\[(UP|DOWN|LEFT|RIGHT)\]', action_content)
            if direction_match:
                direction = direction_match.group(1).lower()
                
                if use_low_instruction:
                    if direction == "up":
                        direction = "down"
                    elif direction == "down":
                        direction = "up"
                    elif direction == "left":
                        direction = "right"
                    elif direction == "right":
                        direction = "left"

                result["to"] = direction
                result["POINT"] = [500, 500]
        elif "LONG_PRESS" in action_content:
            coords_match = re.search(r'\[\[\s*(\d+)\s*,\s*(\d+)\s*\]\]', action_content)
            if coords_match:
                x, y = map(int, coords_match.groups())
                result["POINT"] = [x, y]
                result["duration"] = 1000
        elif "PRESS_BACK" in action_content:
            result["PRESS"] = "BACK"
        elif "PRESS_HOME" in action_content:
            result["PRESS"] = "HOME"
        elif "PRESS_RECENT" in action_content:
            result["PRESS"] = "RECENT"
        elif "WAIT" in action_content:
            result["duration"] = 200
        elif "COMPLETE" in action_content:
            result["STATUS"] = "finish"
        else:
            print(f"Warning: Could not parse a valid action from: {action_content}")
            
    except Exception as e:
        print(f"Error parsing action string '{action_str}': {e}")
        
    return result
def run_episode(episode, image_path, history_list, use_low_instruction):
    if use_low_instruction:
        return run_episode_low(episode, image_path, history_list, use_low_instruction)
    else:
        return run_episode_high(episode, image_path, history_list, use_low_instruction)

def load_image(episode, image_path, history_list, use_low_instruction):
    """
    Checks if the image exists. If not, logs an error and returns None.
    Otherwise, returns the arguments for the prediction task.
    """
    if not os.path.exists(image_path):
        error_dir = os.path.join(current_dir, "atlas_error")
        os.makedirs(error_dir, exist_ok=True)
        error_log_file = os.path.join(error_dir, "missing_files.log")

        error_id = f"{episode.get('episode_id', 'N/A')}_{episode.get('step_id', 'N/A')}"
        error_message = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, FileNotFoundError: {image_path}, id: {error_id}\n"
        
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(error_message)
            
        print(f"Warning: Image {image_path} (id: {error_id}) not found. Skipped and logged.")
        return None
        
    return (episode, image_path, history_list, use_low_instruction)

def predict(args, datasets):
    global USE_LOW_INSTRUCTION
    USE_LOW_INSTRUCTION = args.use_low_instruction
    data_dir = args.data_dir
    split_type = args.split
    print("Predicting on:", datasets)
    
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    
    processed_count = 0
    skipped_count = 0
    error_dir = os.path.join(current_dir, "atlas_error")
    os.makedirs(error_dir, exist_ok=True)

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
                continue
            
            episodes_files = os.listdir(episode_dir)
            
            load_futures = []
            all_tasks = []
            print("Loading episodes")
            with ThreadPoolExecutor(max_workers=16) as executor:
                for episodes_file in episodes_files:
                    episodes_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}.json")
                    try:
                        with open(episodes_path, 'r', encoding='utf-8') as f:
                            episodes = json.load(f)
                    except Exception as e:
                        print(f"Failed to load {episodes_path}: {e}")
                        continue
                    
                    for episode in episodes:
                        episode_history = {
                            "history_instruct": episode.get("history_instruct", [])
                        }
                        episode["category"] = dataset
                        image_path = episode["image_path"]
                        
                        episode_copy = copy.deepcopy(episode)
                        episode_history_copy = copy.deepcopy(episode_history)
                        load_futures.append(executor.submit(load_image, episode_copy, image_path, episode_history_copy, USE_LOW_INSTRUCTION))

                for f in as_completed(load_futures):
                    try:
                        result = f.result()
                        if result is not None:
                            all_tasks.append(result)
                        else:
                            skipped_count += 1
                    except Exception as e:
                        print(f"Error retrieving result from future: {e}")
                        skipped_count += 1

            with open(output_file, "w", encoding="utf-8") as f_out:
                print(f"Predicting {len(all_tasks)} episodes.")

                future_to_info = {}
                for task_value in all_tasks:
                    episode_data = task_value[0]
                    future = poolexec.submit(run_episode, *task_value)
                    info = {
                        "episode_id": episode_data.get("episode_id", "UNKNOWN_ID"),
                        "step_id": episode_data.get("step_id", "UNKNOWN_STEP")
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
                        error_id = f"{episode_info['episode_id']}_{episode_info['step_id']}"
                        error_message = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, Error during prediction task: {e}, id: {error_id}"
                        print(f"\nError: {e}, id: {error_id}")
                        
                        log_file = os.path.join(error_dir, "prediction_errors.log")
                        with open(log_file, 'a', encoding='utf-8') as f_log:
                            f_log.write(error_message + "\n")
                        continue

        print(f"Prediction saved at: {output_file}.")

    os.system(f"cat {args.output_dir}/*/predict.jsonl > {args.output_dir}/all.jsonl")
    print(f"Merged prediction saved at: {args.output_dir}/all.jsonl.")

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Successfully processed episodes: {processed_count}")
    print(f"Skipped episodes due to errors: {skipped_count}")
    print(f"Final evaluation metrics will be based on {processed_count} samples.")
    print("Check 'atlas_error/' directory for details on skipped files.")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OS-Atlas Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/home/test/test03/models/OS-Atlas-Pro-7B"),
                       help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.getenv('OUTPUT_PATH', "eval_results")),
                       help="Directory to save results")
    parser.add_argument("--data_name", type=str, default=os.getenv("PREDICT_DATASET", "GAMBIT"),
                       help="Eval dataset name")
    parser.add_argument("--use_low_instruction", action='store_true', 
                        help='Enable using low-level instructions')
    args = parser.parse_args()
    random.seed(args.seed)

    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    
    print(f'Loading model at : {args.model_path}')
    print(f'Loading data at  : {args.data_dir}')
    print(f'Processing subsets: {data_subset}')
    print(f'Saving results at: {args.output_dir}')
    
    predict(args, data_subset)