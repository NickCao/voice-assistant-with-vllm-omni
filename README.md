# Voice Assistant with vLLM-Omni

Real-time voice assistant powered by [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) served via [vLLM-Omni](https://github.com/vllm-project/vllm-omni), with a [LiveKit](https://livekit.io/) frontend.

## Architecture

```
┌──────────┐  WebRTC  ┌──────────────┐
│ Browser   │◄───────►│ LiveKit      │
│ (React)   │         │ Server :7880 │
│ :3000     │         └──────┬───────┘
└──────────┘                 │
                             ▼
                    ┌──────────────────┐   HTTP POST
                    │ LiveKit Agent    │──────────────────►┌─────────────────────┐
                    │ (Python)         │  /v1/chat/        │ vLLM-Omni           │
                    └──────────────────┘  completions      │ Qwen3-Omni          │
                                                          │ :8091               │
                                                          └─────────────────────┘
```

The browser captures audio via WebRTC, LiveKit routes it to a Python agent. The agent uses Silero VAD for turn detection, then sends the user's audio to vLLM-Omni's chat completion endpoint as a base64-encoded WAV. Qwen3-Omni processes the audio natively (no separate STT/TTS) and returns both text and spoken audio. Conversation history is maintained across turns.

## Prerequisites

- **GPU server:** 2x NVIDIA H100 (or equivalent), [vLLM-Omni](https://github.com/vllm-project/vllm-omni) installed
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
│   └── src/
│       ├── agent.py        # AgentSession entrypoint with Silero VAD
│       └── vllm_realtime.py # RealtimeModel backed by chat completions
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
- Ensure vLLM-Omni is running with `--omni` flag
- Check the GPU server firewall allows port 8091
- Verify: `curl http://<gpu-server-ip>:8091/v1/models`

**No audio response:**
- Check browser microphone permissions
- Verify the agent registered with LiveKit (check agent logs for "registered" message)
- Ensure `LIVEKIT_URL` in frontend `.env.local` matches the LiveKit server address

**High latency:**
- Expected end-to-end latency is ~1-2s on 2x H100
- Check GPU utilization with `nvidia-smi`
