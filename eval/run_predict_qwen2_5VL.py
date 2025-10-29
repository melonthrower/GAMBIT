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
from utils.utils_qwen.agent_function_call import MobileUse
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from utils.qwen_mobile_tool import aitw_2_qwen2_5, aitw_2_qwen2_5_action
import yaml
import time
import torch
from qwen_vl_utils import smart_resize
import random
from yacs.config import CfgNode as CN
import re
import numpy as np
import requests
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.qwen_mobile_tool import aitw_2_uitars
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir

import traceback

def transform_history_string_to_dict(action_list: list) -> dict:
    """
    Transforms a history action from string format (from t.py) to the dict format
    expected by aitw_2_qwen2_5_action.
    Example input: ['TYPE: some text']
    Example output: {'result_action_type': 'type', 'result_action_text': 'some text'}
    """
    if not action_list or not isinstance(action_list, list) or not action_list[0]:
        return {}

    action_str = action_list[0]
    action_dict = {}

    if action_str.startswith("TYPE"):
        text = action_str.split(":", 1)[1].strip()
        action_dict['result_action_type'] = 'type'
        action_dict['result_action_text'] = text
        
    elif action_str.startswith("CLICK"):
        coords_str = action_str.split(":", 1)[1].strip(" ()")
        x1, y1, x2, y2 = map(float, coords_str.split(","))
        touch_x = ((x1 + x2) / 2) / 1000.0
        touch_y = ((y1 + y2) / 2) / 1000.0
        action_dict['result_action_type'] = 'click'
        action_dict['result_touch_yx'] = [touch_y, touch_x]
        action_dict['result_lift_yx'] = [touch_y, touch_x]

    elif action_str.startswith("LONG_PRESS"):
        coords_str = action_str.split(":", 1)[1].strip(" ()")
        x1, y1, x2, y2 = map(float, coords_str.split(","))
        touch_x = ((x1 + x2) / 2) / 1000.0
        touch_y = ((y1 + y2) / 2) / 1000.0
        action_dict['result_action_type'] = 'long_press'
        action_dict['result_touch_yx'] = [touch_y, touch_x]
        action_dict['result_lift_yx'] = [touch_y, touch_x]

    elif action_str.startswith("SCROLL"):
        direction = action_str.split(":", 1)[1].strip().lower()
        action_dict['result_action_type'] = 'scroll' 
        if direction == 'up':
            touch_yx = [0.7, 0.5]; lift_yx = [0.3, 0.5]
        elif direction == 'down':
            touch_yx = [0.3, 0.5]; lift_yx = [0.7, 0.5]
        elif direction == 'left':
            touch_yx = [0.5, 0.7]; lift_yx = [0.5, 0.3]
        else: # right
            touch_yx = [0.5, 0.3]; lift_yx = [0.5, 0.7]
        action_dict['result_touch_yx'] = touch_yx
        action_dict['result_lift_yx'] = lift_yx
            
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

DEVICES = ["cuda:0",
           "cuda:1",
           "cuda:2","cuda:3"
          ]
torch.set_num_threads(4)

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

ERROR_LOG_DIR = os.path.join(current_dir, "qwen2_5vl_error")
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

if current_dir not in sys.path:
    sys.path.append(current_dir)
    
def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)

_llm = None
_tokenizer = None

def _init_llm(model_name):
    global _llm, _tokenizer
    if _llm is None:
        _llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    if _tokenizer is None:
        _tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

def move_to(device):
    global _llm, _tokenizer
    if _llm is None: raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None: raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"

user_query_template = '''The user query:  {user_request} 
Task progress (You have done the following operation on the current device): {history_actions}'''
user_query_template_thought = '''The user query: {user_request}
Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.
After answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.
Task progress (You have done the following operation on the current device):
{history_actions}'''
user_query_template_low = '''The user query:  {user_request} 
Current step query: {low_instruction}
Task progress (You have done the following operation on the current device): {history_actions}'''

def get_qwen_response_base(episode: dict, screenshot_path: str, args=None, is_low_level=False):
    # This function will let errors propagate to the main loop for logging
    global _llm, _tokenizer
    
    user_query = episode["instruction"]
    history_actions = episode.get("history_action", [])

    dummy_image = Image.open(screenshot_path)
    resized_height, resized_width = smart_resize(
        dummy_image.height, dummy_image.width,
        factor=_tokenizer.image_processor.patch_size * _tokenizer.image_processor.merge_size,
        min_pixels=_tokenizer.image_processor.min_pixels,
        max_pixels=_tokenizer.image_processor.max_pixels,
    )
    mobile_use = MobileUse(cfg={"display_width_px": resized_width, "display_height_px": resized_height})

    if history_actions:
        history_actions_str = "".join([
            f"Step {i+1}: {aitw_2_qwen2_5_action(transform_history_string_to_dict(action), resized_height, resized_width).strip()}; " 
            for i, action in enumerate(history_actions)
        ])
    else:
        history_actions_str = "None"

    if is_low_level:
        low_instruction = episode["low_instruction"]
        template = user_query_template_low.format(user_request=user_query, history_actions=history_actions_str, low_instruction=low_instruction)
    else:
        template = user_query_template.format(user_request=user_query, history_actions=history_actions_str)

    prompt = NousFnCallPrompt()
    message = prompt.preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            Message(role="user", content=[
                ContentItem(text=template),
                ContentItem(image=f"file://{screenshot_path}")
            ]),
        ],
        functions=[mobile_use.function],
        lang=None,
    )
    message = [msg.model_dump() for msg in message]
    
    text = _tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # print(text)
    inputs = _tokenizer(text=[text], images=[dummy_image], padding=True, return_tensors="pt")
    device = _llm.device
    inputs = inputs.to(device)

    generation_params = {
        'do_sample': not getattr(args, 'greedy', False), 'top_p': getattr(args, 'top_p', 0.01),
        'top_k': getattr(args, 'top_k', 1), 'temperature': getattr(args, 'temperature', 0.01),
        'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
    }

    output_ids = _llm.generate(**inputs, max_new_tokens=getattr(args, 'out_seq_length', 2048), **generation_params)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    minicpm_answer = qwen2_5_2_minicpm(output_text, resized_height, resized_width)
    episode["raw_pred"] = output_text # Store raw output for debugging
    episode["pred"] = minicpm_answer
    return episode

def qwen2_5_2_minicpm(output_text: str, resized_height: int, resized_width: int) -> dict:
    try:
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL)
        if not tool_call_match:
            print(f"Warning: <tool_call> not found in output: {output_text}")
            return {"STATUS": "FAIL_PARSE"}
        
        action_str = tool_call_match.group(1).strip()
        action = json.loads(action_str)
        qwen_action = action['arguments']
        action_name = qwen_action['action']
        
        if action_name == "click":
            x, y = qwen_action["coordinate"]
            x = x / resized_width * 1000; y = y / resized_height * 1000
            return {"POINT": [int(x), int(y)], "STATUS": "continue"}
        elif action_name == "long_press":
            x, y = qwen_action["coordinate"]
            x = x / resized_width * 1000; y = y / resized_height * 1000
            return {"POINT": [int(x), int(y)], "duration": int(qwen_action.get("time", 1) * 1000), "STATUS": "continue"}
        elif action_name == "swipe":
            x1, y1 = qwen_action["coordinate"]; x2, y2 = qwen_action["coordinate2"]
            x1 = x1 / resized_width * 1000; y1 = y1 / resized_height * 1000
            x2 = x2 / resized_width * 1000; y2 = y2 / resized_height * 1000
            direction = "right" if x2 > x1 else "left"
            if abs(y2 - y1) > abs(x2 - x1):
                direction = "down" if y2 > y1 else "up"
            return {"POINT": [int(x1), int(y1)], "to": direction, "STATUS": "continue"}
        elif action_name == "type":
            return {"TYPE": qwen_action["text"], "STATUS": "continue"}
        elif action_name == "system_button":
            button_map = {"Back": "BACK", "Home": "HOME", "Enter": "ENTER"}
            button = button_map.get(qwen_action["button"])
            return {"PRESS": button, "STATUS": "continue"} if button else {}
        elif action_name == "terminate":
            return {"STATUS": "finish"}
        elif action_name == "wait":
            return {"duration": int(qwen_action.get("time", 1) * 1000), "STATUS": "continue"}
        
        return {"STATUS": "FAIL_PARSE"}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing model output: {e}. Raw output: '{output_text}'")
        return {"STATUS": "FAIL_PARSE"}

def run_episode(episode, image_path, use_low_instruction, args=None):
    return get_qwen_response_base(episode, image_path, args, use_low_instruction)

def load_image(episode, image_path, use_low_instruction):
    return (episode, image_path, use_low_instruction)

def predict(args, datasets):

    use_low_instruction_flag = args.use_low_instruction
    data_dir = args.data_dir
    split_type = args.split
    print("Predicting on:", datasets)
    
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    
    processed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=len(DEVICES), initializer=_init_llm, initargs=(args.model_path,)) as poolexec:
        print("Moving model to devices")
        futures = [poolexec.submit(move_to, dev) for dev in DEVICES]
        for fut in as_completed(futures): print(fut.result())
    
        for dataset in datasets:
            save_dir = os.path.join(args.output_dir, dataset)
            os.makedirs(save_dir, exist_ok=True)
            episode_dir = os.path.join(data_dir, split_type, dataset)
            output_file = os.path.join(save_dir, "predict.jsonl")
            
            if not os.path.exists(episode_dir):
                print(f"Warning: Episode directory not found, skipping: {episode_dir}")
                continue
                
            episodes_files = os.listdir(episode_dir)# Removed slicing for full run
            
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
                        loading_futures.append(executor.submit(load_image, episode_copy, image_path, use_low_instruction_flag))

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
                    future = poolexec.submit(run_episode, *task_value, args)
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

        print(f"Prediction results saved in respective dataset directories under: {args.output_dir}.")
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
    parser = argparse.ArgumentParser(description="Qwen2.5VL Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/share_data/data1/GUI_eval/Qwen2.5-VL-7B-Instruct"),
                       help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.getenv('OUTPUT_PATH', "eval_results")),
                       help="Directory to save results")
    parser.add_argument("--data_name", type=str, default=os.getenv("PREDICT_DATASET", "GAMBIT"),
                       help="Eval dataset name")
    parser.add_argument('--greedy', action='store_true', help='Use greedy decoding')
    parser.add_argument('--top_p', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--out_seq_length', type=int, default=2048)
    parser.add_argument('--use_low_instruction', action='store_true', 
                        help='Enable using low-level instructions.')
    args = parser.parse_args()
    random.seed(args.seed)

    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    
    print(f'Loading model at : {args.model_path}')
    print(f'Loading data at  : {args.data_dir}')
    print(f'Processing subsets: {data_subset}')
    print(f'Saving results at: {args.output_dir}')
    
    predict(args, data_subset)