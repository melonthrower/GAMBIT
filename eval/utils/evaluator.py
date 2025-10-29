import os
import json
import numpy as np
import Levenshtein
import math
from PIL import Image, ImageDraw, ImageFont
from utils.action_type import ActionType
from utils.utils import annotate_and_save_image
from typing import List, Union
from collections import defaultdict
from pprint import pprint
# Based on evaluator of Qwen 2.5 VL
# https://github.com/QwenLM/Qwen2.5-VL/issues/904
# https://gist.github.com/LukeForeverYoung/274a073ca77c9dc46022cb8cc5382223
# https://gist.github.com/LukeForeverYoung/1f5d19495788de0d905c5ac6341153f5

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
schema_dir = os.path.dirname(os.path.dirname(current_file_path))
EXTRACT_SCHEMA = json.load(open(os.path.join(schema_dir, 'utils/schema', 'my_schema_for_extraction.json'), encoding="utf-8"))

# CONSTANTS
_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
_TAP_DISTANCE_THRESHOLD_AC = 0.04  # for android control, align with qwen's code.
_TAP_DISTANCE_THRESHOLD_AC = 40  # for android control, align with qwen's code.
# _SWIPE_DISTANCE_THRESHOLD = 0.04 # Interval determining if an action is a tap or a swipe.
ANNOTATION_WIDTH_AUGMENT_FRACTION= 1.2 # aitw set it to 1.4, aitz and qwen 2.5 vl set it to 1.2.
ANNOTATION_HEIGHT_AUGMENT_FRACTION= 1.2 # We follow qwen setting.
default_duration = EXTRACT_SCHEMA["properties"]["duration"]["default"] # default 200


def _resize_annotation_bounding_boxes(
    annotation_position: Union[List[float], List[List[float]]], ## annotation position可能是列表，可能是列表的列表
    width_factor: float = 1.2,
    height_factor: float = 1.2,
):
    """根据factor放大边界框"""

    def _resize(box: List[float]): ## 返回放大factor后的边界点x,y和h,w
        y, x, h, w = box
        h_delta = (height_factor - 1) * h
        w_delta = (width_factor - 1) * w
        y = max(0, y - h_delta / 2)
        x = max(0, x - w_delta / 2)
        h = min(1, h + h_delta)
        w = min(1, w + w_delta)
        return [y, x, h, w]

    if not annotation_position:
        return []
    if isinstance(annotation_position[0], list):
        return [_resize(b) for b in annotation_position]
    return _resize(annotation_position)


def is_tap_action(normalized_start_yx, normalized_end_yx): 
    '''
    检查是否在标注点附近（通过距离 <= _SWIPE_DISTANCE_THRESHOLD,这里是0.04)
    '''
    distance = np.linalg.norm(np.array(normalized_start_yx) - np.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD


def check_inside(x, y, bbox_list): 
    '''
    检查点是否在bbox_list中
    '''
    
    bbox_array = np.array(bbox_list)
    
    if bbox_array.ndim == 1:
        bbox_array = np.expand_dims(bbox_array, axis=0)
        
    x_min, y_min, x_max, y_max = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3] ## 会有标号反的情况吗

    # Check whether (x, y) is inside any of the bounding boxes
    within_x = (x_min <= x) & (x <= x_max)
    within_y = (y_min <= y) & (y <= y_max)
    within_bbox = within_x & within_y

    if np.any(within_bbox):
        within_bbox_coords = bbox_array[within_bbox]
        return True, within_bbox_coords
    else:
        return False, None


def obtain_gt_bbox(coordinate, bbox_list, eval_android_control=False): ## 需要输入bbox list，返回最近的/最近的五个
    x, y = coordinate['x'], coordinate['y']
    bbox_array = np.array(bbox_list)
    if bbox_array.ndim == 1:
        bbox_array = np.expand_dims(bbox_array, axis=0)
    if len(bbox_list) == 0:
        return []

    if not eval_android_control: 
        is_inside, bbox_inside = check_inside(x, y, bbox_list)
        if is_inside:
            return bbox_inside.tolist()
        else:
            return []
    else: ##
        def get_center_distance(box):
            
            xmin, ymin, xmax, ymax = box
            # print(xmin)
            center_y = ymin + ymax/2
            center_x = xmin + xmax/2
            # print(((center_y - y) ** 2 + (center_x - x) ** 2) ** 0.5)
            return ((center_y - y) ** 2 + (center_x - x) ** 2) ** 0.5

        distances = [get_center_distance(box) for box in bbox_array]
        
        # 找到最小距离及其对应的 bbox
        min_distance = min(distances)
        
        # 核心判断：检查最小距离是否在阈值内
        if min_distance <= _TAP_DISTANCE_THRESHOLD_AC:
            # 如果是，找到那个最近的 bbox 并返回
            min_index = distances.index(min_distance)
            # print(f"min_index = {min_index}")
            return [bbox_list[min_index]]
        else:
            # print(f"min_index = {min_index}")
            # 如果所有 bbox 的中心点都离得太远，则没有匹配项
            return []


def _get_direction(point1, point2): 
    '''
    获得scroll的方向，在up，down，left，right之间选出，是get_direction中调用的函数
    '''
    try:
        x1, y1 = point1["x"], point1["y"]
        x2, y2 = point2["x"], point2["y"]

        assert x1 is not None
        assert x2 is not None
        assert y1 is not None
        assert y2 is not None

        vector = (x2 - x1, y2 - y1)
        vx, vy = vector
    except Exception as e:
        return "no direction"

    directions = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    vector_length = math.sqrt(vx ** 2 + vy ** 2)
    if vector_length == 0:
        return "no direction"
    unit_vector = (vx / vector_length, vy / vector_length)

    max_cosine = -float('inf')
    closest_direction = None
    for direction, dir_vector in directions.items():
        dx, dy = dir_vector
        dir_length = math.sqrt(dx ** 2 + dy ** 2)
        cos_theta = (unit_vector[0] * dx + unit_vector[1] * dy) / dir_length
        if cos_theta > max_cosine:
            max_cosine = cos_theta
            closest_direction = direction

    return closest_direction

def get_direction(point, to): ## 获得方向，按两种方式 1. 已经有UP，Down信息的 2. 还没有设置，只有两点坐标的，调用上面的函数
    if isinstance(to, str):
        if to in ["up", "down", "left", "right"]:
            return to
        else:
            return "no direction"
    elif isinstance(to, list):
        try:
            point1 = {"x": point[0], "y": point[1]}
            point2 = {"x": to[0], "y": to[1]}
            return _get_direction(point1, point2)
        except Exception as e:
            return "no direction"

class ActionEvaluator(object):

    
    def __init__(self, save_dir, eval_android_control=False) -> None:
        self.save_dir = save_dir
        # compatible with aitz evaluator
        self.demo_mode = "COA"
        self.screen_mode = "txt"
        self.annobase = './eval_data/tmp/mobilebenchmark-hard/data/annotations/'
        self._aitz_action_type_ = ActionType
        self._stop_status = [
          "finish",
          "satisfied",
          "impossible",
          "interrupt",
          "need_feedback"
        ]
        self.eval_android_control = eval_android_control
        

    def action_map(self, action_api: dict): ## 返回动作类型，从action_api中的action提取
        action = action_api.get('ACTION', None)
        args = action_api.get('ARGS', None)
        status = action_api.get('STATUS', None)
        duration = args.get('duration', default_duration) if args else None

        if action is None and args is None and status is None:
            print('Schema error. reason: action is None and args is None and status is None')
            return None, {}
        elif status in self._stop_status: ## 如果是结束状态就stop
            return "complete", {}
        elif "impossible" in status:
            return "impossible", {}
        elif "TYPE" in action: ## 输入就返回type和对应的文本
            return "type", action['TYPE']
        elif "POINT" in action and "to" not in args and duration == default_duration: # click point 不带 to 且默认 duration就就是click
            return "click", action['POINT']
        elif "POINT" in action and "to" in args and duration == default_duration: # swipe point 带 to 默认 duration
            return "scroll", {"start": action['POINT'], "end": args['to']}
        elif "POINT" in action and "duration" in args and duration > default_duration: # long press duration比默认长
            return "long_press", {"coordinate": action['POINT'], "duration": args['duration']}
        elif "PRESS" in action:
            if action['PRESS'] == "HOME":
                return "press_home", action['PRESS']
            elif action['PRESS'] == "BACK":
                return "press_back", action['PRESS']
            else:
                return "out of space", {}
        elif "duration" in args: # pause and wait
            return "wait", args['duration']
        elif "wait" in action:
            return "wait",{}
        else:
            raise ValueError("Unknown action type.")


    def _parse_action_(self, pred, image_width=None, image_height=None): 
        '''
        将输出的pred统一到我们的动作空间
        '''
        pd_action_type, pd_action_yx, pd_action_idx, pd_action_text, pd_action_button, pd_action_direction, pd_duration = (None, ) * 7

        # 现在的输入 pred 格式是固定的，直接解析
        pr = pred.get('action_predict', {})
        if self.demo_mode not in pr: return (None, ) * 7

        action = pr[self.demo_mode].get(self.screen_mode, {})
        if not action: return (None, ) * 7
        pd_action_type, pd_action_args = self.action_map(action)
        if pd_action_type is None: print('Unknown action: ', action)

        if pd_action_type == "click":
            try:
                pd_action_yx = {"x": pd_action_args[0], "y": pd_action_args[1]}
            except Exception as e:
                pd_action_yx = {"x": 0.0, "y": 0.0}
        elif pd_action_type == "long_press":
            try:
                pd_action_yx = {"x": pd_action_args["coordinate"][0], "y": pd_action_args["coordinate"][1]}
            except Exception as e:
                pd_action_yx = {"x": 0.0, "y": 0.0}
        else:
            pd_action_yx = None

        pd_action_idx = None

        pd_action_direction = get_direction(pd_action_args.get("start", {}), pd_action_args.get("end", {})) if pd_action_type == "scroll" else None

        pd_action_text = pd_action_args if pd_action_type == "type" else None

        pd_duration = pd_action_args.get("duration") if pd_action_type == "long_press" else None

        return pd_action_type, pd_action_yx, pd_action_idx, pd_action_text, pd_action_button, pd_action_direction, pd_duration

    def _parse_answer_(self, gt):
        actions = []
    
        action_types = gt.get("result_action_type", [])
        action_text = gt.get("result_action_text", "")
        action_directions = gt.get("result_direction", [])
        click_bboxes = gt.get("result_click_bbox", [])
    
        # 各类动作的索引计数器
        click_idx = 0
        direction_idx = 0
    
        for action_type in action_types:
            action = {
                "type": None,
                "text": None,
                "button": None,
                "direction": None,
                "click_box": []
            }
    
            if action_type == "type":
                action["type"] = "type"
                action["text"] = action_text
    
            elif action_type == "scroll":
                action["type"] = "scroll"
                if direction_idx < len(action_directions):
                    action["direction"] = action_directions[direction_idx]
                    direction_idx += 1
    
            elif action_type == "click":
                action["type"] = "click"
                if click_idx < len(click_bboxes):
                    action["click_box"] = click_bboxes[click_idx]
                    click_idx += 1
    
            elif action_type == "long_press":
                action["type"] = "long_press"
                if click_idx < len(click_bboxes):
                    action["click_box"] = click_bboxes[click_idx]
                    click_idx += 1
    
            elif action_type == "press_back":
                action["type"] = "press_back"
                action["button"] = "back"
    
            elif action_type == "press_home":
                action["type"] = "press_home"
                action["button"] = "home"
    
            elif action_type == "impossible":
                action["type"] = "impossible"
    
            elif action_type == "wait":
                action["type"] = "wait"
    
            elif action_type == "complete":
                action["type"] = "complete"
    
            else:
                raise ValueError(f"Unknown action type: {action_type}")
    
            actions.append(action)
    
        return actions
        
    def _calculate_first_error_goal_progress(self, episode_id, episode_steps, annotations_folder_path):
        """
        根据“首次错误”逻辑计算单个 episode 的平均目标完成度。

        Args:
            episode_id (str): 当前 episode 的 ID。
            episode_steps (list): 当前 episode 的所有步骤结果。
            annotations_folder_path (str): 包含标注文件的文件夹路径。

        Returns:
            float: 该 episode 的平均目标完成度 (值在 0.0 到 1.0 之间)。
        """
        # 确保步骤按 step_id_int 排序，如果不存在则使用 step_id
        # 注意: 原始脚本中已经做了排序，但这里为了稳健性再次确认
        annotations_folder_path = self.annobase
        episode_steps.sort(key=lambda x: x.get('step_id_int', int(x.get('step_id', 0))))

        # 1. 尝试从标注文件或步骤数据中获取执行序列
        sequences = None
        episode_json_path = os.path.join(annotations_folder_path, f"{episode_id}.json")

        if os.path.exists(episode_json_path):
            try:
                with open(episode_json_path, 'r', encoding='utf-8') as f:
                    episode_info = json.load(f)
                if 'seq' in episode_info and episode_info['seq']:
                    sequences = list(episode_info['seq'].values())
            except (json.JSONDecodeError, KeyError):
                # 如果文件解析失败或没有 'seq' 键，则忽略
                pass

        # 如果无法从标注文件中获取，则尝试从步骤数据中获取
        if sequences is None:
            if episode_steps and 'atomic_instructions' in episode_steps[0] and episode_steps[0]['atomic_instructions']:
                raw_sequences = list(episode_steps[0]['atomic_instructions'].values())
                sequences = []
                for seq_group in raw_sequences:
                    if seq_group and isinstance(seq_group[0], list):
                         sequences.extend(seq_group)
                    else:
                         sequences.append(seq_group)
        
        # 2. 如果成功获取序列，则计算目标完成度
        if not sequences:
            # 如果没有找到序列信息，无法计算，返回0.0
            return 0.0

        sequence_progress_list = []
        for seq in sequences:
            if not seq: continue
            try:
                # 确保序列中的步骤ID是整数
                int_seq = [int(s) for s in seq]
            except (ValueError, TypeError):
                continue # 如果序列格式不正确，则跳过

            total_steps_in_seq = len(int_seq)
            if total_steps_in_seq == 0: continue

            steps_before_first_error = total_steps_in_seq
            for i, step_index in enumerate(int_seq):
                # 查找对应的步骤结果
                current_step = next((s for s in episode_steps if s.get('step_id_int') == step_index), None)
                # 如果步骤未找到，或该步骤的 exact_match 不为 True，则认为这是第一个错误点
                if not current_step or not current_step.get('eval_result', {}).get('exact_match', False):
                    steps_before_first_error = i
                    break
            
            progress = steps_before_first_error / total_steps_in_seq
            sequence_progress_list.append(progress)

        if sequence_progress_list:
            average_progress = sum(sequence_progress_list) / len(sequence_progress_list)
            return average_progress  # 返回浮点数 (例如 0.75)
        
        # 如果所有序列都无法计算，则返回0.0
        return 0.0

    def compute_episode_metrics(self, episode_results):
        """
        计算按类别划分的成功率和目标完成度。
        目标完成度使用“首次错误”逻辑进行计算。

        Args:
            episode_results (dict): { "episode_key": [step_results] }
            annotations_folder_path (str): 包含 {id}.json 标注文件的文件夹路径。
        """
        # episode_results is a dict: { "episode_key": [step_results] }
        annotations_folder_path = self.annobase
        # Stores success status (True/False) for each category
        success_by_category = defaultdict(list)
        # Stores goal progress (float) for each category
        gp_by_category = defaultdict(list)
    
        for episode_key, episode_steps in episode_results.items():
            if not episode_steps:
                continue
            
            # Get category from the first step (it's the same for all steps in an episode)
            category = episode_steps[0].get("category", "unknown")
            
            # Calculate success (task completion) - 这部分逻辑保持不变
            is_success = all(step.get("eval_result", {}).get("exact_match", False) for step in episode_steps)
            success_by_category[category].append(is_success)
    
            # --- GOAL PROGRESS CALCULATION (MODIFIED) ---
            # 原有的 max_progress 逻辑已被移除。
            # 调用新的辅助方法来计算目标完成度。
            goal_progress_value = self._calculate_first_error_goal_progress(
                episode_id=episode_key,
                episode_steps=episode_steps,
                annotations_folder_path=annotations_folder_path
            )
            gp_by_category[category].append(goal_progress_value)
    
        # --- Aggregate results (这部分逻辑保持不变) ---
        output_metrics = {
            "by_category": {},
            "overall": {}
        }
        
        all_successes = []
        all_gps = []
    
        for category in success_by_category:
            successes = success_by_category[category]
            gps = gp_by_category[category]
            
            # 注意：这里的 gps 已经是 0.0 到 1.0 的浮点数
            sr = (sum(successes) / len(successes)) * 100 if successes else 0
            avg_gp = (sum(gps) / len(gps)) * 100 if gps else 0
            
            output_metrics["by_category"][category] = {
                "success_rate": round(sr, 2),
                "goal_progress": round(avg_gp, 2),
                "episode_count": len(successes)
            }
            all_successes.extend(successes)
            all_gps.extend(gps)
    
        # Calculate overall metrics
        overall_sr = (sum(all_successes) / len(all_successes)) * 100 if all_successes else 0
        overall_gp = (sum(all_gps) / len(all_gps)) * 100 if all_gps else 0
        
        output_metrics["overall"] = {
            "success_rate": round(overall_sr, 2),
            "goal_progress": round(overall_gp, 2),
            "episode_count": len(all_successes)
        }
    
        # For backward compatibility, return top-level keys
        output_metrics["success_rate"] = output_metrics["overall"]["success_rate"]
        output_metrics["goal_progress"] = output_metrics["overall"]["goal_progress"]
    
        return output_metrics

    def __call__(self, gt, pred, annotate_image=False): ## gt是traj里的每一步，pred是results里面的各个步
        """ eval_single_step """
        pd_action_detail = None
        pixel_distance = None

        image_width, image_height = gt['image_width'], gt['image_height']

        subset, episode_id, step_id, task_desc = gt['subset'], gt['episode_id'], gt['step_id'], gt['instruction']
        
        gt_actions = self._parse_answer_(gt)

        # get predict action information
        pd_action_type, pd_action_yx, pd_action_idx, \
            pd_action_text, pd_action_button, pd_action_direction, pd_duration = self._parse_action_(pred, image_width, image_height)
        pd_action_detail={
            "click": pd_action_yx,
            "scroll": pd_action_direction,
            "type": pd_action_text,
            "long_press": pd_action_yx,
            "press_home": "home",
            "press_back": "back",
            "complete": "complete",
            "impossible": "impossible",
            "wait": "wait",
            "out_of_space": "out_of_space"
        }.get(pd_action_type, None)

        type_match = False
        exact_match = False
        text_dist = None
        hit_format = pd_action_type is not None
            
        matched_gt_action = None

        # 4. Loop through all GT actions to find a match, if a prediction was made
        if pd_action_type is not None:
            for gt_action in gt_actions:
                gt_action_type = gt_action['type']
                
                # Check for type match first
                if pd_action_type == gt_action_type:
                    type_match = True
                    current_exact_match = False

                    # Perform exact match check based on type
                    if pd_action_type in ["click", "long_press"]:
                        gt_click_box = gt_action.get('click_box', [])
                        if pd_action_detail and gt_click_box:
                            gt_bbox_found = obtain_gt_bbox(pd_action_detail, gt_click_box, self.eval_android_control)
                            if gt_bbox_found:
                                current_exact_match = True
                    
                    elif pd_action_type == "scroll":
                        gt_direction = gt_action.get('direction')
                        if  pd_action_detail and gt_direction and pd_action_detail.lower() == gt_direction.lower():
                            current_exact_match = True

                    elif pd_action_type == "type":
                        gt_text = gt_action.get('text', "")
                        if pd_action_detail is not None and gt_text is not None:
                            pd_text_norm = pd_action_detail.lower().strip()
                            gt_text_norm = gt_text.lower().strip()
                            if not text_dist: # Calculate only once
                                text_dist = Levenshtein.ratio(pd_text_norm, gt_text_norm)
                            if pd_text_norm in gt_text_norm or gt_text_norm in pd_text_norm:
                                current_exact_match = True

                    elif pd_action_type in ["press_home", "press_back", "complete", "impossible", "wait"]:
                        current_exact_match = True

                    # If an exact match is found with this GT action, we are done.
                    if current_exact_match:
                        exact_match = True
                        matched_gt_action = gt_action # Store the action we matched with
                        break # Exit the loop as we found a valid match

            # If a type match was found but no exact match, still record the first gt_action of that type for reference
            if type_match and not exact_match:
                # if( gt_action['type'] == 'click'):
                #     print(f"gt_action")
                # print(1)
                for gt_action in gt_actions:
                    if gt_action['type'] == pd_action_type:
                        matched_gt_action = gt_action
                        break

        # 5. Log if no exact match was found
        if not exact_match:
            all_gt_details_for_log = [(a['type'], a.get('click_box') or a.get('direction') or a.get('text') or a['type']) for a in gt_actions]
            # match_type_str = "No Type Match" if not type_match else "No Exact Match"
            # print(f"\n{match_type_str}, pd action: {pd_action_type}, detail: {pd_action_detail}; gt actions: {all_gt_details_for_log}, {subset}_{episode_id}_{step_id}")

        # 6. Prepare final result dictionary
        # If a match was found, use its details. Otherwise, use the first GT action as a fallback for reporting.
        final_gt_action = matched_gt_action or (gt_actions[0] if gt_actions else None)
        final_gt_type = final_gt_action['type'] if final_gt_action else "N/A"
        final_gt_detail = None
        if final_gt_action:
            final_gt_detail = {
                "click": final_gt_action.get('click_box'), "long_press": final_gt_action.get('click_box'),
                "scroll": final_gt_action.get('direction'), "type": final_gt_action.get('text'),
            }.get(final_gt_type, final_gt_type)
        is_success = exact_match and pd_action_type  in ["complete", "impossible"]
        return {
            "subset": subset, "episode_id": episode_id, "step_id": step_id,
            "answer": {"action_type": final_gt_type, "action_detail": final_gt_detail},
            "pred": {"action_type": pd_action_type, "action_detail": pd_action_detail},
            "type_match": type_match, "exact_match": exact_match,
            "text_dist": text_dist, "format_hit": hit_format, "pixel_distance": pixel_distance,
            "success": is_success,
        }

    @staticmethod
    def compute_atomic_instruction_metrics(results):
        """Computes atomic instruction accuracy, broken down by category."""
        atomic_step_correctness = defaultdict(lambda: {'correctness': [], 'category': 'unknown'})
        
        for step_result in results:
            # ... (same logic as before to find the atomic instance key)
            is_correct = step_result.get("eval_result", {}).get("exact_match", 0) == True
            step_id = step_result.get("step_id")
            episode_key = f"{step_result.get('subset')}-{step_result.get('episode_id')}"
            atomic_info = step_result.get("atomic_instructions", {})
            
            found = False
            for task_name, sequences in atomic_info.items():
                for seq_idx, step_sequence in enumerate(sequences):
                    if step_id in step_sequence:
                        atomic_instance_key = f"{episode_key}-{task_name}-{seq_idx}"
                        atomic_step_correctness[atomic_instance_key]['correctness'].append(is_correct)
                        
                        atomic_step_correctness[atomic_instance_key]['category'] = step_result.get("category", "unknown")
                        found = True
                        break
                if found:
                    break
        # print("--- [调试信息] atomic_step_correctness 字典内容 (格式化输出) ---")
        # # 我们将 defaultdict 转换为普通 dict 进行打印，这样输出更纯净
        # pprint(dict(atomic_step_correctness))
        # print("--- [调试信息] 字典内容结束 ---")
        # =======================================================

# --- Aggregate results by category ---
        stats_by_category = defaultdict(lambda: {"correct": 0, "total": 0})
        # --- Aggregate results by category ---
        stats_by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    
        for instance_key, data in atomic_step_correctness.items():
            if not data['correctness']: continue
            
            category = data['category']
            is_instance_correct = all(data['correctness'])
            
            stats_by_category[category]['total'] += 1
            if is_instance_correct:
                stats_by_category[category]['correct'] += 1
    
        output_metrics = {
            "by_category": {},
            "overall": {}
        }
        total_correct = 0
        total_all = 0
    
        for category, stats in stats_by_category.items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            output_metrics["by_category"][category] = {
                "atomic_instruction_accuracy": round(accuracy, 2),
                "correct_atomic_instructions": stats['correct'],
                "total_atomic_instructions": stats['total']
            }
            total_correct += stats['correct']
            total_all += stats['total']
    
        overall_accuracy = (total_correct / total_all) * 100 if total_all > 0 else 0
        output_metrics["overall"] = {
            "atomic_instruction_accuracy": round(overall_accuracy, 2),
            "correct_atomic_instructions": total_correct,
            "total_atomic_instructions": total_all
        }
    
        # For backward compatibility
        output_metrics["atomic_instruction_accuracy"] = output_metrics["overall"]["atomic_instruction_accuracy"]
        return output_metrics
    
    @staticmethod
    def compute_atomic_metrics(step_results):
        # 算各类动作的准确率
        recorder = {
            'total':  {'count':0, 'type_match':0, 'exact_match':0, "hit": 0},
            # -------------------------------------------
            'CLICK':      {'count':0, 'type_match':0, 'exact_match':0},
            'TYPE':       {'count':0, 'type_match':0, 'exact_match':0, 'text_dist': []},
            'SCROLL':     {'count':0, 'type_match':0, 'exact_match':0},
            'PRESS':      {'count':0, 'type_match':0, 'exact_match':0},
            'STOP':       {'count':0, 'type_match':0, 'exact_match':0},
            'LONG_PRESS': {'count':0, 'type_match':0, 'exact_match':0},
        }
        for step in step_results:
            recorder['total']['count'] += 1
            recorder['total']['hit'] += step.get('format_hit', 0)

            gt_action_type = step.get('answer', {}).get('action_type')
            if not isinstance(gt_action_type, str):
                continue
            
            # --- 新增的映射逻辑 ---
            # 将具体的动作类型映射到 recorder 的分类键
            category = None
            if gt_action_type == 'click':
                category = 'CLICK'
            elif gt_action_type == 'long_press':
                category = 'LONG_PRESS'
            elif gt_action_type == 'type':
                category = 'TYPE'
            elif gt_action_type == 'scroll':
                category = 'SCROLL'
            elif gt_action_type in ['press_home', 'press_back']:
                category = 'PRESS'
            elif gt_action_type in ['complete', 'impossible', 'wait']:
                category = 'STOP'
            # --- 映射逻辑结束 ---

            if category and category in recorder:
                recorder[category]['count'] += 1
                recorder[category]['type_match'] += step.get('type_match', 0)
                recorder['total']['type_match'] += step.get('type_match', 0)
                recorder[category]['exact_match'] += step.get('exact_match', 0)
                recorder['total']['exact_match'] += step.get('exact_match', 0)
                if 'text_dist' in recorder[category] and step.get('text_dist') is not None:
                    recorder[category]['text_dist'].append(step['text_dist'])

        # Initialize scores dictionary, including counts and ratios
        scores = {
            metric_key: {
                'count': recorder[metric_key]['count'],
                'type_acc': round(
                    (recorder[metric_key]['type_match'] / recorder[metric_key]['count']) * 100, 2
                ) if recorder[metric_key]['count'] > 0 else 0,
                'exact_acc': round(
                    (recorder[metric_key]['exact_match'] / recorder[metric_key]['count']) * 100, 2
                ) if recorder[metric_key]['count'] > 0 else 0
            }
            for metric_key in ['total', 'CLICK', 'LONG_PRESS', 'SCROLL', 'PRESS', 'STOP', 'TYPE']
        }

        # Calculate hit_rate
        scores['total']['hit_rate'] = round(
            (recorder['total']['hit'] / recorder['total']['count']) * 100, 2
        ) if recorder['total']['count'] > 0 else 0

        # Calculate average text_dist for TYPE
        if recorder['TYPE']['text_dist']:
            scores['TYPE']['text_dist_avg'] = round(
                sum(recorder['TYPE']['text_dist']) / len(recorder['TYPE']['text_dist']), 4
            )
        else:
            scores['TYPE']['text_dist_avg'] = 0

        # Calculate pixel distance
        pixel_distances = [
            step['pixel_distance'] for step in step_results
            if step.get('pixel_distance') is not None
        ]

        median_pixel_distance = round(
            float(np.median(pixel_distances)), 4
        ) if pixel_distances else -1

        mean_pixel_distance = -1

        if pixel_distances:
            pixel_distances = np.array(pixel_distances)
            filtered_distances = pixel_distances[pixel_distances < 1e15]
            if len(filtered_distances) > 0:
                mean_pixel_distance = round(
                    float(np.mean(filtered_distances)), 4
                )

        scores['mean_pixel_distance'] = mean_pixel_distance
        scores['median_pixel_distance'] = median_pixel_distance

        return scores