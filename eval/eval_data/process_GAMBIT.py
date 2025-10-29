import os
import json
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
split_data_path = os.path.join(current_dir, 'tmp/GAMBIT/data/test_anno/random_split.json')

out_dir = Path(os.path.join(current_dir, 'GAMBIT/test/GAMBIT'))
out_dir.mkdir(parents=True, exist_ok=True)

def transform_action_data(action_data:dict) -> dict:

    # directly retrieve relevant information from the data.
    episode_id:str     = action_data["image"].split('/')[-1].split("_")[0]
    step_id:str        = action_data["image"].split('/')[-1].split("_")[1][:-4]
    episode_length:str = action_data["step_length"]
    image_path:str     = action_data["image"]
    instruction:str    = action_data["question"]
    step_instruction:str         = action_data["current_instruction"]
    history_action = action_data["history_action"]
    history_screenshot = action_data["history_screenshot"]
    history_instruct = action_data['history_instruct_list']
    atomic_instructions = action_data.get("atomic_instructions", {})
    step_id_int = action_data.get("step_id_int", -1) 
    category = action_data.get("category", "unknown")
    seq_info = action_data.get("seq", {})
    answers = action_data["answer"] if isinstance(action_data["answer"], list) else [action_data["answer"]] ## 有多解的情况需要变成列表

    # Get the picture information (w/h) of the data. They are restored under annotation/*.json.
    with open(os.path.join(current_dir, "tmp/GAMBIT/data/annotations", f"{episode_id}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    image_width:int = int(data["w"])
    image_height:int = int(data["h"])

    
    results = []
    result_action_type = []
    result_direction = []
    result_click_bbox = []
    result_action_text = ""

    result_touch_yx = []
    result_lift_yx = []
    
    for answer in answers:
        # first intialize the variables.
        
        if answer.startswith("CLICK:"):
            nums = list(map(float, answer.split(":")[1].strip(" ()").split(",")))
            result_action_type.append("click")
            x1, y1, x2, y2 = nums
            if image_width > 0 and image_height > 0:
                x1_scaled = round((x1 / image_width) * 1000)
                y1_scaled = round((y1 / image_height) * 1000)
                x2_scaled = round((x2 / image_width) * 1000)
                y2_scaled = round((y2 / image_height) * 1000)
                scaled_bbox = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
            else:
                scaled_bbox = [x1, y1, x2, y2]
            
            result_click_bbox.append(scaled_bbox)
            result_touch_yx = [round((y1_scaled+y2_scaled)/2),round((x1_scaled+x2_scaled)/2)]
            result_lift_yx = [round((y1_scaled+y2_scaled)/2),round((x1_scaled+x2_scaled)/2)]

        if answer.startswith("SCROLL"):
            result_action_type.append("scroll")
            result_direction.append(answer[8:])
            tmp = [500,500]
            result_touch_yx.append(deepcopy(tmp))
            if answer.endswith("UP"):
                tmp[0] -= 50
            elif answer.endswith("DOWN"):
                tmp[0] += 50
            elif answer.endswith("LEFT"):
                tmp[1] -= 50
            else:
                tmp[1] += 50
            result_lift_yx.append(tmp)

        if answer.startswith("LONG_PRESS"):
            result_action_type.append("long_press")
            x1,y1,x2,y2 = list(map(float, answer.split(":")[1].strip(" ()").split(",")))
            if image_width > 0 and image_height > 0:
                x1_scaled = round((x1 / image_width) * 1000)
                y1_scaled = round((y1 / image_height) * 1000)
                x2_scaled = round((x2 / image_width) * 1000)
                y2_scaled = round((y2 / image_height) * 1000)
                scaled_bbox = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
            else:
                scaled_bbox = [x1, y1, x2, y2]
            result_click_bbox.append(scaled_bbox)
            result_touch_yx = [round((y1_scaled+y2_scaled)/2),round((x1_scaled+x2_scaled)/2)]
            result_lift_yx = [round((y1_scaled+y2_scaled)/2),round((x1_scaled+x2_scaled)/2)]

        if answer.startswith("TYPE"):
            result_action_type.append("type")
            result_action_text:str = answer[5:].strip()

        if answer.startswith("PRESS_HOME"):
            result_action_type.append("press_home")

        if answer.startswith("PRESS_BACK"):
            result_action_type.append("press_back")

        if answer.startswith("COMPLETE"):
            result_action_type.append("complete")

        if answer.startswith("IMPOSSIBLE"):
            result_action_type.append("impossible")

        if answer.startswith("WAIT"):
            result_action_type.append("wait")

    data = {
        "episode_id": episode_id,
        "step_id": step_id,
        "episode_length": episode_length,
        "image_width": image_width,
        "image_height": image_height,
        "image_path": image_path,
        "instruction": instruction,
        "result_action_type": result_action_type,
        "result_action_text": result_action_text,
        "result_click_bbox": result_click_bbox,
        "result_direction": result_direction,
        "ui_positions": "",
        "low_instruction": step_instruction,
        "history_action": history_action,
        "history_screenshot": history_screenshot,
        "history_instruct": history_instruct,
        "atomic_instructions": atomic_instructions,
        "step_id_int": step_id_int,
        "category": category,
        "seq": seq_info,
        "result_touch_yx": result_touch_yx,
        "result_lift_yx": result_lift_yx,
    }

    return data

# Construct the data.
with open(split_data_path, "r", encoding="utf-8") as f:
    eval_data_raw:dict = json.load(f)

data_eval = [transform_action_data(data) for data in tqdm(eval_data_raw)]
data_eval = [d for d in data_eval if d is not None]

# save data
def dump_traj(traj, out_root: Path, idx: int):
    if not traj:
        return

    subfolder_name = f"traj_{idx:05d}"
    subfolder_path = out_root / subfolder_name
    subfolder_path.mkdir(parents=True, exist_ok=True)

    out_filename = f"{subfolder_name}.json"
    out_file = subfolder_path / out_filename

    with out_file.open("w", encoding="utf-8") as fw:
        json.dump(traj, fw, ensure_ascii=False, indent=2)

    print(f"Save {subfolder_name}/{out_filename}  (steps={len(traj)})")


records = data_eval
traj = []
traj_idx = 1
prev_step_id, curr_instr = None, None

for i, rec in enumerate(records):
    try:
        step_id = int(rec["step_id"])
    except KeyError as e: 
        print("-" * 50)
        print(f"Processing failed! An error occurred at the {i}-th record in the records list.")
        print(f"Error message: A KeyError was caught, and the missing key is {e}")
        print(f"The complete content of the record causing the error is: \n{rec}")
        print("-" * 50)
        continue
    instr = rec["instruction"]
    rec['subset'] = 'GAMBIT'
    rec['step_id'] = int(rec['step_id'])

    new_traj = (
        curr_instr is None or
        instr != curr_instr or
        prev_step_id is None or
        step_id != prev_step_id + 1
    )

    if new_traj and traj:
        dump_traj(traj, out_dir, traj_idx)
        traj_idx += 1
        traj = []

    traj.append(rec)
    prev_step_id = step_id
    curr_instr   = instr

dump_traj(traj, out_dir, traj_idx)
print("all done.")