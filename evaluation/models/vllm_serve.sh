# MODEL_NAME="Qwen/Qwen3-8B"
MODEL_NAME="Qwen/Qwen3-14B"
PORT=1209
DEVICE=2,3

CUDA_VISIBLE_DEVICES=$DEVICE python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $PORT \
    --tensor-parallel-size 2