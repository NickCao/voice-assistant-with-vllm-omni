# vLLM-Omni Realtime API — OpenAI Compatibility Analysis

vLLM-Omni's `/v1/realtime` WebSocket endpoint is based on vLLM's speech-to-text realtime handler, not a full implementation of the OpenAI Realtime API. This document tracks the known differences.

## 1. `session.update` Format Mismatch

**OpenAI format:**
```json
{"type": "session.update", "session": {"model": "...", "audio": {...}}}
```

**vLLM-Omni expects:**
```json
{"type": "session.update", "model": "..."}
```

The `model` field must be at the top level, not nested inside `session`. This breaks LiveKit's `livekit-plugins-openai` which sends the OpenAI format.

**Workaround applied:** Patched `vllm/entrypoints/speech_to_text/realtime/connection.py` line 106 to also check `event["session"]["model"]`.

## 2. Error Response Format

**OpenAI returns:**
```json
{"type": "error", "error": {"message": "...", "code": "..."}}
```

**vLLM-Omni returns:**
```json
{"type": "error", "error": "...", "code": "..."}
```

The `error` field is a plain string instead of an object with a `message` attribute. This causes the LiveKit OpenAI plugin to crash with `AttributeError: 'str' object has no attribute 'message'`.

## 3. Supported Events (Subset)

**vLLM-Omni handles:**
- `session.update` — configure model
- `input_audio_buffer.append` — send audio chunk
- `input_audio_buffer.commit` — trigger transcription/generation

**OpenAI Realtime API also supports (not implemented in vLLM-Omni):**
- `response.create` — explicitly request a response
- `conversation.item.create` — add items to conversation context
- `input_audio_buffer.clear` — clear the audio buffer
- Turn detection configuration events
- Tool call / function calling events

The LiveKit plugin sends additional `session.update` events for instructions and tool configurations that vLLM-Omni may silently ignore or reject.

## 4. Output Events Differ

**vLLM-Omni emits:**
- `transcription.delta` — streaming partial text
- `transcription.done` — final text with usage stats
- `response.audio.delta` — incremental audio output (PCM16, base64)
- `response.audio.done` — signals end of audio

**OpenAI Realtime API emits:**
- `response.audio_transcript.delta` — streaming text transcript
- `response.audio.delta` — incremental audio
- `response.done` — response complete
- `conversation.item.created` — conversation lifecycle events
- `session.created` / `session.updated` — session lifecycle

## 5. Patches Applied to This Deployment

### Patch 1: FlashInfer Sampler Disabled
**File:** `vllm/v1/sample/ops/topk_topp_sampler.py` (line 39)

Changed `if envs.VLLM_USE_FLASHINFER_SAMPLER:` to `if False:` because the cloudexe environment is missing CUDA development headers (`curand.h`) needed for FlashInfer JIT compilation.

### Patch 2: Model Field in session.update
**File:** `vllm/entrypoints/speech_to_text/realtime/connection.py` (line 106)

Changed:
```python
model = event.get("model")
```
To:
```python
model = event.get("model") or (event.get("session") or {}).get("model")
```

This allows the LiveKit OpenAI plugin's `session.update` format (with model nested inside `session`) to be accepted.

## 6. Mid-Generation Disconnect Crash

vLLM-Omni crashes if the WebSocket client disconnects while audio generation is in progress. The server does not gracefully handle cancellation of active generation requests when the connection drops. This is a server-side bug — clients should implement clean shutdown by waiting for `response.audio.done` before closing, or accepting that the server may crash on abrupt disconnection.

## Summary

vLLM-Omni's realtime endpoint is **OpenAI-inspired but not OpenAI-compatible**. It works for simple audio-in → audio-out flows but does not implement the full session/conversation/response lifecycle that OpenAI's Realtime API (and clients like LiveKit's plugin) expect. To work around these issues, this project uses a custom `VLLMRealtimeModel` bridge (`agent/src/vllm_realtime.py`) that speaks vLLM-Omni's protocol natively instead of going through the LiveKit OpenAI plugin.
