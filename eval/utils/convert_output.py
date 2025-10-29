import json
import os
import jsonschema
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 假设 EXTRACT_SCHEMA 的路径是正确的
# Get the absolute path of the current file
try:
    current_file_path = os.path.abspath(__file__)
    schema_dir = os.path.dirname(os.path.dirname(current_file_path))
    EXTRACT_SCHEMA = json.load(open(os.path.join(schema_dir, 'utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))
except FileNotFoundError:
    print("Warning: schema_for_extraction.json not found. Continuing without schema validation.")
    EXTRACT_SCHEMA = None


def load_json_data(file_path):
    data = []
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode line in jsonl file: {line}")
    return data


def parse_action(data): ## 输入pred, 返回status，action_keys
    try:
        jsonschema.validate(data, EXTRACT_SCHEMA)
        
        actions = {}
        parameters = {}
        status = data.get("STATUS", "continue")  # Default value

        # Define actions
        action_keys = ["POINT", "to", "PRESS", "TYPE"]
        
        # Extract actions
        for key in action_keys:
            if key in data:
                actions[key] = data[key]
        
        # Extract global parameters
        parameters["duration"] = data.get("duration", EXTRACT_SCHEMA["properties"]["duration"]["default"])

        # Handle "to" parameter, if present
        if "to" in data:
            parameters["to"] = data["to"]
            
        return actions, parameters, status

    except Exception as e:
        print('Error, JSON is NOT valid according to the schema.')
        return None, None, None


def process_step(args):
    """Processes a single step item and saves it to a file."""
    step_item, base_path = args
    
    task = step_item.get("category", step_item.get("subset", "unknown"))
    episode_id = step_item.get("episode_id", "unknown")
    step_id = step_item.get("step_id", "unknown")
    pred_data = step_item.get("pred", {})
    
    actions, parameters, status = parse_action(pred_data)
    
    if actions is None:
        return False

    # This is the crucial part: create a dictionary that contains BOTH
    # the converted prediction AND the metadata needed for evaluation.
    transformed_entry = {
        "action_predict": {
            "COA": {
                "txt": {
                    "ACTION": actions,
                    "ARGS": parameters,
                    "STATUS": status
                },
            }
        },
        # Carry over all necessary metadata from the original step item
        "subset": task,
        "category": task,
        "episode_id": episode_id,
        "step_id": step_id,
        "atomic_instructions": step_item.get("atomic_instructions", {}),
    }

    folder = f"{task}-{episode_id}"
    file_name = f"{folder}_{step_id}.json"
    output_file_path = os.path.join(base_path, folder, file_name)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(transformed_entry, output_file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving file {output_file_path}: {e}")
        return False


def convert2aitz(input_path, output_path, max_workers=None):
    """Main function to convert model output to AITZ format for evaluation."""
    data = load_json_data(input_path)
    if not data:
        print("No data loaded from input file.")
        return

    base_path = os.path.join(output_path)
    folders_to_create = set()
    tasks_to_submit = []

    # Prepare tasks and identify needed directories
    for item in data:
        task = item.get("category", item.get("subset", "unknown"))
        episode_id = item.get("episode_id", "unknown")
        
        folder = f"{task}-{episode_id}"
        folders_to_create.add(folder)
        tasks_to_submit.append((item, base_path))
    
    # Create all necessary subdirectories
    for folder in folders_to_create:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    
    # Use multiprocessing to process all steps
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_results = executor.map(process_step, tasks_to_submit)
        for result in tqdm(future_results, total=len(tasks_to_submit), desc="Converting to AITZ format"):
            if result:
                success_count += 1
    
    print(f"Conversion complete. {success_count} steps processed successfully.")