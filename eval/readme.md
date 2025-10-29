# Evaluation Scripts

Except for infigui-r1, our evaluation scripts are largely based on the evaluation scripts in the official AgentCPM repository. Please refer to the [corresponding repository](https://github.com/OpenBMB/AgentCPM-GUI/tree/main/eval) for more details.

# Data preparation
For models other than infigui-R1, our data preprocessing follows the following process:
Download [GAMBIT](https://huggingface.co/datasets/melonthrower12138/GAMBIT) here, then save and unzip it in eval_data/tmp/GAMBIT.
Run preprocess.py in eval_data/tmp/GAMBIT then preprocess_GAMBIT.py in eval_data

For infigui-R1, after downloading the GAMBIT dataset, use the following command to generate the processed data.
```
python format_convert_GAMBIT2ac.py
```

# Inference
## AgentCPM-GUI-8B
```
## If you want to enable low-instruction, add '--use_low_instruction' to the command
python run_predict_minicpm.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/AgentCPM-GUI/GAMBIT --data_name GAMBIT
```
## Qwen2.5-VL-7B
```
## If you want to enable low-instruction, add '--use_low_instruction' to the command
python run_predict_qwen2_5VL.py --model_path ../model/Qwen2.5-VL-7B-Instruct --output_dir ./eval_results/Qwen2.5-VL-7B-Instruct/GAMBIT --data_name GAMBIT
```
## UI-TARS-7B-SFT
```
## If you want to enable low-instruction, add '--use_low_instruction' to the command
python run_predict_ui_tars.py --model_path ../model/UI-TARS-7B-SFT --output_dir ./eval_results/UI-TARS-7B-SFT/GAMBIT --data_name GAMBIT
```
## OS-Atlas-Pro-7B
```
## If you want to enable low-instruction, add '--use_low_instruction' to the command
python run_predict_os_atlas.py --model_path ../model/OS-Atlas-Pro-7B --output_dir ./eval_results/OS-Atlas-Pro-7B/GAMBIT --data_name GAMBIT
```
## Aguvis-7B-720P
```
## If you want to enable low-instruction, add '--use_low_instruction' to the command
python run_predict_aguvis.py --model_path ../model/aguvis --output_dir ./eval_results/aguvis/GAMBIT --data_name GAMBIT
```

## Infigui-R1
We modified the script in the official repository used for evaluating Android control to evaluate our dataset.
```
## Set 'eval_type' to 'low' or 'high' in the command to determine whether to use low_instruction.
## thinking
python android_control.py --model_path {model_path} --eval_type low --thinking --output_dir ./eval_result/{output_file_name} 
## no-thinking
python android_control.py --eval_type low --output_dir ./eval_result/{output_file_name}
```
# Eval
After executing the corresponding inference script, run the following eval script. Replace the model name below with the corresponding model name.

```
python run_eval_agent.py --input_path ./eval_results/{ModelName}/GAMBIT/all.jsonl --output_dir ./eval_results/{ModelName}/GAMBIT/results --data_name GAMBIT
```

Please run the statistical analysis scripts after executing the corresponding inference script and the run_eval_agent.py script.

We provide statistical analysis code for the following evaluation metrics:
```
analyze_action.py: Calculates statistics based on the type of action.
analyze_atomic_goal_progress.py: Calculates the Weighted Longest Common Subsequence.
analyze_branch_accuracy.py: Calculates the branch frame accuracy.
analyze_results_by_type.py: Calculates statistics based on the instruction type.
```

calculate_metrics.py: The statistics script for infigui.

