#!/bin/bash
# Start vLLM-Omni serving Qwen3-Omni on 2x GPUs.
# Run this on the GPU server (2x H100).
# Install: pip install vllm-omni
set -e
exec vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8091
