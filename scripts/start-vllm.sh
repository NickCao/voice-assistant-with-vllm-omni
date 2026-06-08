#!/bin/bash
# Start vLLM-Omni serving Qwen3-Omni on 2x GPUs.
# Run this on the GPU server (2x H100).
# The --omni flag handles multi-GPU stage distribution automatically
# (Thinker on GPU 0, Talker+Code2Wav on GPU 1).
# Do NOT use --tensor-parallel-size with --omni.
# Install: pip install vllm-omni
set -e
exec vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni \
    --host 0.0.0.0 \
    --port 8091
