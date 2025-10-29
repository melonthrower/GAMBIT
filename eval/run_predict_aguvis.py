#!/usr/bin/env python3
# coding: utf-8
"""
Inference code is modified from the origin repo: https://github.com/xlang-ai/aguvis
See the origin code at:
Inference: https://github.com/xlang-ai/aguvis/blob/main/src/aguvis/serve/cli.py
Prompts: https://github.com/xlang-ai/aguvis/blob/main/src/aguvis/constants.py
"""
import torch
import os
import re
import sys
import json
import argparse
import random
import warnings
import multiprocessing
from tqdm import tqdm
from io import BytesIO
from typing import List, Literal, Optional, Dict
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor

import time
import traceback


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if current_dir not in sys.path:
    sys.path.append(current_dir)

ERROR_LOG_DIR = os.path.join(current_dir, "aguvis_error")
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

# Ignore the warnings of eos_end_token.
warnings.filterwarnings("ignore")

import requests
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import NoneType
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

import logging
logging.getLogger("transformers").setLevel(logging.WARNING)

from utils.utils import get_dataset_dir

DEVICES = [
    "cuda:0","cuda:1","cuda:2","cuda:3"
    ]

# Define prompt settings =============
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."
# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
# System Message
grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
# Chat Template
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
assistant_template = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% if 'recipient' in message %}<|recipient|>{{ message['recipient'] }}{% endif %}{{'\n' + message['content'][0]['text']}}{% if 'end_turn' in message and message['end_turn'] %}{{'<|diff_marker|>\n'}}{% else %}{{'<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|recipient|>' }}{% endif %}"
# Special Tokens
additional_special_tokens = ["<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<|recipient|>", "<|diff_marker|>",]
# Plugin Functions
select_option_func = {"name": "browser.select_option", "description": "Select an option from a dropdown menu", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the dropdown menu"}, "y": {"type": "number", "description": "The y coordinate of the dropdown menu"}, "value": {"type": "string", "description": "The value of the option to select"}}, "required": ["x", "y", "value"]}}
swipe_func = {"name": "mobile.swipe", "description": "Swipe on the screen", "parameters": {"type": "object", "properties": {"from_coord": {"type": "array", "items": {"type": "number"}, "description": "The starting coordinates of the swipe"}, "to_coord": {"type": "array", "items": {"type": "number"}, "description": "The ending coordinates of the swipe"}}, "required": ["from_coord", "to_coord"]}}
home_func = {"name": "mobile.home", "description": "Press the home button"}
back_func = {"name": "mobile.back", "description": "Press the back button"}
wait_func = {"name": "mobile.wait", "description": "wait for the change to happen", "parameters": {"type": "object", "properties": {"seconds": {"type": "number", "description": "The seconds to wait"}}, "required": ["seconds"]}}
terminate_func = {"name": "mobile.terminate","description": "Terminate the task when it is completed or cannot be continued.","parameters": {"type": "object","properties": {"status": {"type": "string","description": "Set to 'success' if the task is completed successfully, or 'failure' if the task has failed and cannot be recovered.","enum": ["success", "failure"]}},"required": ["status"]}
}
long_press_func = {"name": "mobile.long_press", "description": "Long press on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the long press"}, "y": {"type": "number", "description": "The y coordinate of the long press"}}, "required": ["x", "y"]}}
open_app_func = {"name": "mobile.open_app", "description": "Open an app on the device", "parameters": {"type": "object", "properties": {"app_name": {"type": "string", "description": "The name of the app to open"}}, "required": ["app_name"]}}
agent_system_message = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nYou have access to the following functions:\n- {json.dumps(swipe_func)}\n- {json.dumps(home_func)}\n- {json.dumps(back_func)}\n- {json.dumps(wait_func)}\n- {json.dumps(long_press_func)}\n- {json.dumps(terminate_func)}"""
user_instruction = """Please generate the next move according to the ui screenshot, instruction and previous actions.\n\nInstruction: {overall_goal}\n\nPrevious actions:\n{previous_actions}"""
until = ["<|diff_marker|>"]
# Prompt setting ends. =============

# action mapping begins. ===========
def mapping_actions(episode: dict) -> dict:
    FAIL_PARSE = {"STATUS": "FAIL_PARSE"}
    action_str = episode.get("pred", "")
    if not isinstance(action_str, str) or not action_str.strip():
        episode["pred"] = FAIL_PARSE
        return episode

    action = action_str.split('\n')[-1].strip()
    platform_match = re.match(r"(\w+)\.", action)
    if not platform_match:
        print(f'Unrecognized output format: {repr(episode["pred"])}.')
        episode["pred"] = FAIL_PARSE
        return episode

    platform = platform_match.group(1)
    function = action[len(platform) + 1 :]

    if platform == "pyautogui":
        if function.startswith("click"):
            try:
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", function)
                x,y = matches
                x = round(float(x) * 1000)
                y = round(float(y) * 1000)
                episode["pred"] = {"POINT": [x, y], "duration": 200, "STATUS": "continue"}
            except Exception as e:
                print(f"Failed to parse POINT ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE
        elif function.startswith("write"):
            try:
                pattern = r'message=(["\'])(.*?)\1'
                match = re.search(pattern, function)
                text = match.group(2)
                episode["pred"] = {"TYPE": text, "duration": 200, "STATUS": "continue"}
            except Exception as e:
                print(f"Failed to parse TYPE ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE
        elif function.startswith("scroll"):
            try:
                pattern = r'scroll\(page=([-+]?\d*\.\d+|\d+)\)'
                match = re.match(pattern, function)
                value = float(match.group(1))
                episode["pred"] = {"POINT": [500, 500], "to": "up" if value < 0 else "down", "duration": 200, "STATUS": "continue"}
            except Exception as e:
                print(f"Failed to parse SCROLL ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE
        elif function.startswith("hscroll"):
            try:
                pattern = r'hscroll\(page=([-+]?\d*\.\d+|\d+)\)'
                match = re.match(pattern, function)
                value = float(match.group(1))
                episode["pred"] = {"POINT": [500, 500], "to": "left" if value < 0 else "right", "duration": 200, "STATUS": "continue"}
            except Exception as e:
                print(f"Failed to parse HSCROLL ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE
        else:
            print(f"Unrecognized action in {platform}: {function}")
            episode["pred"] = FAIL_PARSE
    elif platform == "mobile":
        if function.startswith("back"):
            episode["pred"] = {"PRESS": "BACK", "duration": 200, "STATUS": "continue"}
        elif function.startswith("home"):
            episode["pred"] = {"PRESS": "HOME", "duration": 200, "STATUS": "continue"}
        elif function.startswith("terminate"):
            if 'success' in action:
                episode["pred"] = {"STATUS": "finish"}
            else:
                episode["pred"] = {"STATUS": "interrupt"}
        elif function.startswith("open_app"):
            try:
                match = re.search(r"app_name='([^']+)'", function)
                app_name = match.group(1)
                episode["pred"] = {"open_app": app_name, "duration": 200, "STATUS": "continue"}
            except Exception as e:
                print(f"Failed to parse open_app ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE
        elif function.startswith("wait"):
            episode["pred"] = {"duration": 3000, "STATUS": "continue"}
        elif function.startswith("long_press"):
            try:
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", action)
                x,y = matches
                x = round(float(x) * 1000)
                y = round(float(y) * 1000)
                episode["pred"] = {"POINT": [x, y], "duration": 1000, "STATUS": "continue"}
            except Exception as e:
                print(f"Failed to parse LONG_PRESS ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE
        else:
            print(f"Unrecognized action in {platform}: {function}")
            episode["pred"] = FAIL_PARSE
    else:
        print(f'Unrecognized platform: {platform}. Full action: {repr(episode["pred"])}.')
        episode["pred"] = FAIL_PARSE
    return episode

_llm: Optional[Qwen2VLForConditionalGeneration] = None
_processor: Optional[Qwen2VLProcessor] = None
_tokenizer = None

def _init_llm(model_name:str) -> None:
    global _llm, _processor, _tokenizer
    if _llm is None:
        _llm = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2", local_files_only=False) # Changed to False for robustness
    if _processor is None:
        _processor = Qwen2VLProcessor.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    if _tokenizer is None:
        _tokenizer = _processor.tokenizer

def move_to(device):
    global _llm,_tokenizer, _processor
    if _llm is None: raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _processor is None: raise ValueError("Error, Processor is not initialized.")
    if _tokenizer is None: raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved model to {device}"

def process_data(episode:Dict, image_path: str, args:argparse.Namespace) -> Optional[Dict]:
    def load_image(image_file: str) -> Image:
        if image_file.startswith(("http://", "https://")):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    try:
        image: Image = load_image(image_path)
    except FileNotFoundError:
        return None # Return None if image not found to skip this task
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    instruction:str = episode['instruction']
    low_instruction: Optional[str] = None
    data_name:str = args.data_name

    if args.use_low_instruction:
        low_instruction:str = episode.get('low_instruction') # Use .get for safety

    history_actions = episode.get("history_action", [])

    return {
        "image": image,
        "episode": episode,
        "instruction": instruction,
        "previous_actions": "",
        "low_level_instruction": low_instruction,
        "mode": args.mode,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens
    }

def generate_response(image: Image.Image, episode: Dict, instruction: str, previous_actions: Optional[str | List[str]] = None, low_level_instruction: Optional[str] = None, mode: Literal["self-plan", "force-plan", "grounding"] = "self-plan", temperature: float = 0, max_new_tokens: int = 1024) -> str:
    global _llm, _processor, _tokenizer
    system_message = {"role": "system", "content": grounding_system_message if mode == "grounding" else agent_system_message}
    if isinstance(previous_actions, list):
        formatted_actions = []
        for action_step in previous_actions:
            if isinstance(action_step, list):
                formatted_actions.append("; ".join(action_step))
            elif isinstance(action_step, str):
                formatted_actions.append(action_step)
        previous_actions = formatted_actions
    if isinstance(previous_actions, list):
        previous_actions = "\n".join(previous_actions)
    if not previous_actions:
        previous_actions = "None"

    user_message = {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_instruction.format(overall_goal=instruction, previous_actions=previous_actions)}]}

    if low_level_instruction:
        recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
    elif mode == "grounding":
        recipient_text = "<|im_start|>assistant<|recipient|>os\n"
    elif mode == "self-plan":
        recipient_text = "<|im_start|>assistant<|recipient|>"
    elif mode == "force-plan":
        recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
    else:
        raise ValueError(f"Invalid mode: {mode}")

    messages = [system_message, user_message]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template)
    text += recipient_text
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(_llm.device)
    cont = _llm.generate(**inputs, do_sample=False, temperature=temperature, max_new_tokens=max_new_tokens)
    cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
    text_outputs = _tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
    for term in until:
        if term:
            text_outputs = text_outputs.split(term)[0]
    episode["pred"] = text_outputs
    return episode

def predict(args:argparse.Namespace):
    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    print(f"Predicting on: {args.data_dir}/{args.split}")
    print(f"Data subset: {data_subset}")

    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    processed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=len(DEVICES),initializer=_init_llm,initargs=(args.model_path,)) as poolexec:
        print("Moving model to devices")
        futures = [poolexec.submit(move_to, dev) for dev in DEVICES]
        for fut in as_completed(futures): print(fut.result())

        for dataset in data_subset:
            save_dir = os.path.join(args.output_dir, dataset)
            os.makedirs(save_dir, exist_ok=True)
            episode_dir = os.path.join(args.data_dir, args.split, dataset)
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
                        
                        loading_futures.append(executor.submit(process_data, episode, image_path, args))

                for f in as_completed(loading_futures):
                    try:
                        result = f.result()
                        if result:
                            all_tasks_to_submit.append(result)
                        else:
                            skipped_count += 1
                    except Exception as exc:
                        print(f'An exception occurred during task loading: {exc}')
                        skipped_count += 1
                        
            all_tasks_to_submit = all_tasks_to_submit
            with open(output_file, "w", encoding="utf-8") as f_out:
                print(f"Predicting {len(all_tasks_to_submit)} episodes for {dataset}...")
                
                future_to_info = {}
                for task_value in all_tasks_to_submit:
                    episode_data = task_value["episode"]
                    future = poolexec.submit(generate_response, **task_value)
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
                        episode = mapping_actions(episode)
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
            print(f"Prediction saved at: {output_file}.")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--data_name", type=str, default="GAMBIT", help="Eval dataset name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, choices = ["self-plan", "force-plan", "grounding"], default="self-plan")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--use_low_instruction", action='store_true', help="If set, use low-level instructions for the model.")
    args = parser.parse_args()
    random.seed(args.seed)
    if args.model_path and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    args.model_path = os.path.abspath(args.model_path)
    print(f'Loading model at : {args.model_path}')
    print(f'Saving results at: {args.output_dir}')
    predict(args)