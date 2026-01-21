import os
from openai import OpenAI

OPENAI_API_KEY = "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"

def gpt_generate(prompt, model_name="gpt-5.2"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    chat_response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
    response = chat_response.choices[0].message.content.strip()
    reasoning = ""
    return response, reasoning

# Evaluation
prompt_template = """
You are a clinical reasoning evaluator.

Task:
Given a patient profile and a list of candidate diseases, determine the single most likely final diagnosis.

Rules (must follow):
1) Choose EXACTLY ONE diagnosis from the provided Candidate Diseases list.
2) Do NOT invent new diseases or use synonyms not appearing in the list. Output the diagnosis name EXACTLY as written.
3) Use ONLY information in the Patient Profile. Do not assume missing facts.
4) If multiple candidates seem plausible, choose the one with the strongest overall support (most specific matching findings + least contradictions).

Output:
Return ONLY a valid JSON object (no markdown, no extra text) that matches this schema:
{{
  "reasoning": "find evidence from the patient profile to support the diagnosis.",
  "final_diagnosis": "the disease name that is most likely to be the final diagnosis"
}}

Patient Profile:
{current_case_prompt}

Candidate Diseases:
{current_choices}
"""

import random
random.seed(42)
import json
from tqdm import tqdm

with open("./processed_data/evaluation_data.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)
evaluation_data = evaluation_data[:100]

def parse_response(response):
    try:
        if isinstance(response, (list, tuple)):
            for part in response:
                if isinstance(part, str) and part.strip():
                    response = part
                    break
        if not isinstance(response, str):
            return response
        if "```json" in response:
            response = response.split("```json", 1)[1].split("```", 1)[0]
        return json.loads(response)
    except Exception:
        return response

def evaluate_each_case(item, k):
    case = item["case_prompt"]
    groundtruth_diagnosis = item["groundtruth_diagnosis"]
    candidates = item["candidates"]
    choices = random.choices(candidates, k=k)
    choices.append(groundtruth_diagnosis)
    random.shuffle(choices)
    prompt = prompt_template.format(current_case_prompt=case, current_choices="\n".join(choices))
    response = gpt_generate(prompt, model_name="gpt-5.2")
    extracted_response = parse_response(response)
    predicted_diagnosis = None
    if type(extracted_response) == dict and "final_diagnosis" in extracted_response:
        predicted_diagnosis = extracted_response["final_diagnosis"]
        if predicted_diagnosis == groundtruth_diagnosis:
            accuracy = 1
        else:
            accuracy = 0
    else:
        accuracy = 0
    return {
        "prompt": prompt,
        "accuracy": accuracy,
        "response": extracted_response,
        "predicted_diagnosis": predicted_diagnosis,
        "groundtruth_diagnosis": groundtruth_diagnosis,
        "candidates": choices,
    }

ks = [4, 9, 49, 99, 499]
for k in ks:
    print(f"Evaluating with k={k}")
    results = []
    for item in tqdm(evaluation_data):
        current_data = item
        result = evaluate_each_case(item, k)
        results.append(result)
        with open(f"results/evaluation_results_{k}.json", "w") as f:
            json.dump(results, f)