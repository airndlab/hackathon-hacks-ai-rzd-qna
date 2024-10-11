#!/bin/bash

# Запуск vLLM сервера в фоновом режиме
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --gpu_memory_utilization 0.8 \
    --quantization awq \
    > vllm.log 2>&1 &

# Запуск FastAPI приложения
uvicorn app.main:app --host 0.0.0.0 --port 8080
