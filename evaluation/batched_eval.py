import os
import json
import random

import yaml
from tqdm import tqdm
from pathlib import Path

from models.batched_generate import batch_generate


def _parse_response(response):
    try:
        if "</think>" in response:
            response = response.split("</think>")[1]
        if "```json" in response:
            response = response.split("```json")[1]
            response = response.split("```")[0]
            response = response.strip()
        response = json.loads(response)
        return response
    except Exception as e:
        return response

def _load_prompt_template(name="zeroshot"):
    base_dir = Path(__file__).resolve().parent
    prompt_path = base_dir / "prompts" / f"{name}.yml"
    prompt_yaml = yaml.safe_load(prompt_path.read_text())
    return prompt_yaml["prompt_template"]

def prepare_data_zeroshot(evaluation_data_path, k=199):
    prompt_template = _load_prompt_template("zeroshot")
    with open(evaluation_data_path, "r") as f:
        evaluation_data = json.load(f)
    all_instances = []
    for instance in evaluation_data:
        case = instance["case_prompt"]
        groundtruth_diagnosis = instance["groundtruth_diagnosis"]
        candidates = instance["candidates"]
        choices = random.choices(candidates, k=k)
        choices.append(groundtruth_diagnosis)
        random.shuffle(choices)
        prompt = prompt_template.format(current_case_prompt=case, current_choices="\n".join(choices))
        all_instances.append({
            "prompt": prompt,
            "groundtruth_diagnosis": groundtruth_diagnosis,
            "candidates": choices,
        })
    return all_instances

def prepare_data_zeroshot_w_description(evaluation_data_path, k=199):
    prompt_template = _load_prompt_template("zeroshot_w_description")
    with open(evaluation_data_path, "r") as f:
        evaluation_data = json.load(f)
    all_instances = []
    for instance in evaluation_data:
        case = instance["case_prompt"]
        groundtruth_diagnosis = instance["groundtruth_diagnosis"]
        candidates = instance["candidates"]
        choices = random.choices(candidates, k=k)
        description_payload = instance.get("description", {})
        candidate_descriptions = description_payload.get("candidate_descriptions", [])
        gt_description = description_payload.get("gt_description", "")
        choices_with_description = [
            f"{choice} (Description: {description})"
            for choice, description in zip(choices, candidate_descriptions)
        ]
        if gt_description:
            choices_with_description.append(
                f"{groundtruth_diagnosis} (Description: {gt_description})"
            )
        random.shuffle(choices_with_description)
        prompt = prompt_template.format(current_case_prompt=case, current_choices="\n".join(choices_with_description))
        all_instances.append({
            "prompt": prompt,
            "groundtruth_diagnosis": groundtruth_diagnosis,
            "candidates": choices_with_description,
        })
    return all_instances

def evaluate_zeroshot(evaluation_data_path = "./data/MedCaseReasoning/processed_data/evaluation_data.json", k=199, model_name="Qwen/Qwen3-8B", num_gpu=4):
    all_instances = prepare_data_zeroshot(evaluation_data_path, k)
    all_prompts = [instance["prompt"] for instance in all_instances]
    prompts, responses = batch_generate(all_prompts, model_name, num_gpu)
    results = []
    for instance, prompt, response in tqdm(zip(all_instances, all_prompts, responses), desc="Evaluating Zeroshot (K={k})"):
        groundtruth_diagnosis = instance["groundtruth_diagnosis"]
        candidates = instance["candidates"]
        extracted_response = _parse_response(response)
        predicted_diagnosis = None
        if type(extracted_response) == dict and "final_diagnosis" in extracted_response:
            predicted_diagnosis = extracted_response["final_diagnosis"]
            if predicted_diagnosis == groundtruth_diagnosis:
                accuracy = 1
            else:
                accuracy = 0
        else:
            accuracy = 0
        results.append({
            "prompt": prompt,
            "accuracy": accuracy,
            "response": extracted_response,
            "predicted_diagnosis": predicted_diagnosis,
            "groundtruth_diagnosis": groundtruth_diagnosis,
            "candidates": candidates,
        })
    return results

def evaluate_zeroshot_w_description(evaluation_data_path = "./data/MedCaseReasoning/processed_data/evaluation_data.json", k=199, model_name="Qwen/Qwen3-8B", num_gpu=4):
    all_instances = prepare_data_zeroshot_w_description(evaluation_data_path, k)
    all_prompts = [instance["prompt"] for instance in all_instances]
    prompts, responses = batch_generate(all_prompts, model_name, num_gpu)
    results = []
    for instance, prompt, response in tqdm(zip(all_instances, all_prompts, responses), desc="Evaluating Zeroshot with Description (K={k})"):
        groundtruth_diagnosis = instance["groundtruth_diagnosis"]
        candidates = instance["candidates"]
        extracted_response = _parse_response(response)
        predicted_diagnosis = None
        if type(extracted_response) == dict and "final_diagnosis" in extracted_response:
            predicted_diagnosis = extracted_response["final_diagnosis"]
            if predicted_diagnosis == groundtruth_diagnosis:
                accuracy = 1
            else:
                accuracy = 0
        else:
            accuracy = 0
        results.append({
            "prompt": prompt,
            "accuracy": accuracy,
            "response": extracted_response,
            "predicted_diagnosis": predicted_diagnosis,
            "groundtruth_diagnosis": groundtruth_diagnosis,
            "candidates": candidates,
        })
    return results

eval_data_path = "/u/bli16/workspace/sea/data/MedCaseReasoning/processed_data/evaluation_data.json"
ks = [199, 59, 9]
models = ["Qwen/Qwen3-8B"]

# result_dir = "./results/zeroshot"
# os.makedirs(result_dir, exist_ok=True)

# for model in models:
#     for k in ks:
#         results = evaluate_zeroshot(eval_data_path, k, model, num_gpu=4)
#         with open(os.path.join(result_dir, f"{model.replace('/', '_')}_{k}.json"), "w") as f:
#             json.dump(results, f, indent=4)

result_dir = "./results/zeroshot_w_description"
os.makedirs(result_dir, exist_ok=True)
for model in models:
    for k in ks:
        results = evaluate_zeroshot_w_description(eval_data_path, k, model, num_gpu=4)
        with open(os.path.join(result_dir, f"{model.replace('/', '_')}_{k}.json"), "w") as f:
            json.dump(results, f, indent=4)