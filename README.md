# Voice Assistant with vLLM-Omni

Real-time voice assistant powered by [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) served via [vLLM-Omni](https://github.com/vllm-project/vllm-omni), with a [LiveKit](https://livekit.io/) frontend.

## Architecture

```
[Local machine]                                           [GPU server (2x H100)]
┌──────────┐  WebRTC  ┌──────────────┐                    ┌─────────────────────┐
│ Browser   │◄───────►│ LiveKit      │                    │ vLLM-Omni           │
│ (React)   │         │ Server :7880 │                    │ Qwen3-Omni          │
│ :3000     │         └──────┬───────┘                    │ --tp 2              │
└──────────┘                 │                            │ :8091               │
                             ▼                            └──────────┬──────────┘
                    ┌──────────────────┐   WebSocket                │
                    │ LiveKit Agent    │◄──────────────────────────►│
                    │ (Python)         │  ws://<gpu-host>:8091/v1/realtime
                    └──────────────────┘
```

The browser captures audio via WebRTC, LiveKit routes it to a Python agent, and the agent streams it to vLLM-Omni's realtime WebSocket endpoint. Qwen3-Omni processes the audio natively (no separate STT/TTS) and streams spoken responses back.

## Prerequisites

- **GPU server:** 2x NVIDIA H100, [vLLM-Omni](https://github.com/vllm-project/vllm-omni) installed
- **Local machine:** Python 3.10+, Node.js 18+, pnpm, [livekit-server](https://github.com/livekit/livekit/releases)

Install LiveKit server (macOS):

```bash
brew install livekit
```

## Setup

### 1. GPU Server — Start vLLM-Omni

```bash
./scripts/start-vllm.sh
```

Or manually:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8091
```

Wait for "Application startup complete" before proceeding. Verify:

```bash
curl http://<gpu-server-ip>:8091/v1/models
```

### 2. Local Machine — Configure Environment

```bash
cp .env.example agent/.env.local
cp .env.example frontend/.env.local
```

Edit both `.env.local` files and set `VLLM_BASE_URL` to your GPU server:

```
VLLM_BASE_URL=http://<gpu-server-ip>:8091/v1
```

### 3. Local Machine — Start LiveKit Server

```bash
./scripts/start-livekit.sh
```

Dev mode uses API key `devkey` and secret `secret` (matching `.env.example` defaults).

### 4. Local Machine — Start the Agent

```bash
./scripts/start-agent.sh
```

This creates a virtual environment on first run, installs dependencies, and starts the agent in dev mode. You should see it register with the LiveKit server.

### 5. Local Machine — Start the Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:3000, click **Start Conversation**, and speak.

## Project Structure

```
├── agent/                  # LiveKit Python agent
│   ├── pyproject.toml
│   └── src/agent.py        # Connects to vLLM-Omni via RealtimeModel
├── frontend/               # Next.js web UI
│   ├── app/
│   │   ├── api/token/      # JWT token generation for LiveKit
│   │   └── page.tsx
│   └── components/
│       └── VoiceAssistant.tsx
└── scripts/                # Startup scripts
```

## Configuration

All configuration is via environment variables in `.env.local` files:

| Variable | Default | Description |
|---|---|---|
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | `devkey` | LiveKit API key |
| `LIVEKIT_API_SECRET` | `secret` | LiveKit API secret |
| `VLLM_BASE_URL` | `http://localhost:8091/v1` | vLLM-Omni HTTP endpoint |

## Troubleshooting

**Agent can't connect to vLLM-Omni:**
- Ensure vLLM-Omni is running with `--omni` flag (required for `/v1/realtime` endpoint)
- Check the GPU server firewall allows port 8091
- Enable debug logging: `LK_OPENAI_DEBUG=1 python src/agent.py dev`

**No audio response:**
- Check browser microphone permissions
- Verify the agent registered with LiveKit (check agent logs for "registered" message)
- Ensure `LIVEKIT_URL` in frontend `.env.local` matches the LiveKit server address

**High latency:**
- Expected end-to-end latency is ~500ms-1s on 2x H100
- Ensure tensor parallelism is enabled (`--tensor-parallel-size 2`)
- Check GPU utilization with `nvidia-smi`
