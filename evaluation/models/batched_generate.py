import os
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def batch_generate(prompts, model_name="gpt-5", num_gpus=1):
    if "gpt" in model_name.lower():
        return gpt_batch_generate(prompts, model_name)
    else:
        return vllm_batch_generate(prompts, model_name, num_gpus)

def gpt_batch_generate(prompts, model_name="gpt-5"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chat_response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt} for prompt in prompts])
    responses = [chat_response.choices[0].message.content.strip() for chat_response in chat_response]
    return responses

def vllm_batch_generate(prompts, model_name="Qwen/Qwen3-8B", num_gpus=1):
    llm = LLM(model=model_name, tensor_parallel_size=num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_token_length = tokenizer.model_max_length
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_token_length)
    messages_list = [
        [{"role": "user", "content": prompt}]
        for prompt in prompts
    ]
    texts = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate(texts, sampling_params)
    prompts = [output.prompt for output in outputs]
    responses = [output.outputs[0].text for output in outputs]
    return prompts, responses
    
if __name__ == "__main__":
    import os
    import json
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    prompts = ["What is the capital of France?", "What is the capital of Germany?"]
    # models = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-Thinking-2507", "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-32B-Think", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
    models = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
    for model in models:
        prompts, responses = batch_generate(prompts, model_name=model, num_gpus=2)
        print(f"Model: {model}")
        print(f"Responses: {responses}")
        with open(f"responses_{model.replace('/', '_')}.json", "w") as f:
            json.dump(responses, f)
        print("-"*100)