from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator
from typing import Literal
from urllib.parse import urlencode, urlparse

import aiohttp
from livekit import rtc
from livekit.agents.llm.chat_context import ChatContext, ChatItem
from livekit.agents.llm.realtime import (
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice, ToolContext
from livekit.agents.types import NOT_GIVEN, NotGivenOr

logger = logging.getLogger("vllm-realtime")

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class _AudioStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()

    def push(self, frame: rtc.AudioFrame) -> None:
        self._queue.put_nowait(frame)

    def close(self) -> None:
        self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame


class _TextStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    def push(self, text: str) -> None:
        self._queue.put_nowait(text)

    def close(self) -> None:
        self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        text = await self._queue.get()
        if text is None:
            raise StopAsyncIteration
        return text


class _MessageStream:
    def __init__(self, generation: MessageGeneration) -> None:
        self._generation = generation
        self._sent = False

    def __aiter__(self) -> AsyncIterator[MessageGeneration]:
        return self

    async def __anext__(self) -> MessageGeneration:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return self._generation


class _EmptyFunctionStream:
    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class VLLMRealtimeModel(RealtimeModel):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        sample_rate: int = OUTPUT_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )
        self._base_url = base_url
        self._model_name = model
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._sessions: set[VLLMRealtimeSession] = set()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return urlparse(self._base_url).hostname or "vllm-omni"

    def _build_ws_url(self) -> str:
        parsed = urlparse(self._base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        if not path.endswith("/realtime"):
            path += "/realtime"
        query = urlencode({"model": self._model_name})
        return f"{scheme}://{parsed.hostname}:{parsed.port}{path}?{query}"

    def session(self) -> VLLMRealtimeSession:
        sess = VLLMRealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        for sess in list(self._sessions):
            await sess.aclose()
        self._sessions.clear()


class VLLMRealtimeSession(RealtimeSession):
    def __init__(self, realtime_model: VLLMRealtimeModel) -> None:
        super().__init__(realtime_model)
        self._model = realtime_model
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._send_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._connected = False
        self._closed = False
        self._main_task: asyncio.Task | None = None
        self._chat_ctx = ChatContext()
        self._tool_ctx = ToolContext([])
        self._instructions = ""
        self._current_audio_stream: _AudioStream | None = None
        self._current_text_stream: _TextStream | None = None
        self._generation_future: asyncio.Future | None = None
        self._full_transcript = ""
        self._audio_chunks_sent = 0
        self._resampler: rtc.AudioResampler | None = None

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._tool_ctx

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tool_ctx = ToolContext(tools)

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        self._ensure_connected()
        pcm_bytes = bytes(frame.data)
        audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
        self._audio_chunks_sent += 1
        if self._audio_chunks_sent % 100 == 1:
            logger.debug(
                "Sending audio chunk #%d (%d bytes, %dHz)",
                self._audio_chunks_sent, len(pcm_bytes), frame.sample_rate,
            )
        self._send_queue.put_nowait({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        self._ensure_connected()
        self.commit_audio()
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.get_event_loop().create_future()
        self._generation_future = fut
        return fut

    def commit_audio(self) -> None:
        logger.info("Committing audio buffer (%d chunks sent)", self._audio_chunks_sent)
        self._send_queue.put_nowait({
            "type": "input_audio_buffer.commit",
        })

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        logger.info("Interrupt requested — closing current generation")
        self._finish_generation()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    def _ensure_connected(self) -> None:
        if self._main_task is None and not self._closed:
            self._main_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        ws_url = self._model._build_ws_url()
        logger.info("Connecting to vLLM-Omni realtime at %s", ws_url)

        headers = {}
        if self._model._api_key:
            headers["Authorization"] = f"Bearer {self._model._api_key}"

        start_time = time.time()
        try:
            self._http_session = aiohttp.ClientSession()
            self._ws = await self._http_session.ws_connect(ws_url, headers=headers)
            acquire_time = time.time() - start_time
            self._connected = True
            self._report_connection_acquired(acquire_time)
            logger.info("Connected to vLLM-Omni realtime (%.2fs)", acquire_time)

            self._send_queue.put_nowait({
                "type": "session.update",
                "model": self._model._model_name,
            })

            send_task = asyncio.create_task(self._send_loop())
            recv_task = asyncio.create_task(self._recv_loop())

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                if task.exception():
                    logger.error("Task failed: %s", task.exception())

        except Exception as e:
            logger.error("Failed to connect to vLLM-Omni: %s", e)
            self.emit(
                "error",
                RealtimeModelError(
                    timestamp=time.time(),
                    label="vllm-omni",
                    error=e,
                    recoverable=False,
                ),
            )
        finally:
            self._connected = False
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._http_session and not self._http_session.closed:
                await self._http_session.close()

    async def _send_loop(self) -> None:
        while self._connected and self._ws and not self._ws.closed:
            try:
                event = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
                await self._ws.send_str(json.dumps(event))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Send error: %s", e)
                break

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                    await self._handle_event(event)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from vLLM-Omni: %s", msg.data[:200])
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                logger.info("WebSocket closed: %s", msg.type)
                break

    async def _handle_event(self, event: dict) -> None:
        event_type = event.get("type", "")

        if event_type == "session.created":
            logger.info("vLLM-Omni session created")

        elif event_type == "transcription.delta":
            delta = event.get("delta", "")
            if delta:
                self._full_transcript += delta
                if self._current_text_stream:
                    self._current_text_stream.push(delta)

        elif event_type == "transcription.done":
            text = event.get("text", self._full_transcript)
            logger.info("Transcription done: %s", text[:100])
            self.emit(
                "input_audio_transcription_completed",
                InputTranscriptionCompleted(
                    item_id="vllm-transcript",
                    transcript=text,
                    is_final=True,
                ),
            )

        elif event_type == "response.audio.delta":
            audio_b64 = event.get("audio", "")
            if not audio_b64:
                return

            sample_rate = event.get("sample_rate_hz", OUTPUT_SAMPLE_RATE)
            pcm_bytes = base64.b64decode(audio_b64)
            samples_per_channel = len(pcm_bytes) // 2  # PCM16 = 2 bytes per sample

            if samples_per_channel == 0:
                return

            frame = rtc.AudioFrame(
                data=pcm_bytes,
                sample_rate=sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_channel,
            )

            if self._current_audio_stream is None:
                self._start_generation()

            if self._current_audio_stream:
                self._current_audio_stream.push(frame)

        elif event_type == "response.audio.done":
            logger.info("Audio response complete")
            self._finish_generation()

        elif event_type == "error":
            error_msg = event.get("error", "Unknown error")
            code = event.get("code", "unknown")
            logger.error("vLLM-Omni error [%s]: %s", code, error_msg)
            self.emit(
                "error",
                RealtimeModelError(
                    timestamp=time.time(),
                    label="vllm-omni",
                    error=Exception(f"[{code}] {error_msg}"),
                    recoverable=True,
                ),
            )

        else:
            logger.debug("Unhandled vLLM-Omni event: %s", event_type)

    def _start_generation(self) -> None:
        logger.info("Starting generation — creating audio/text streams")
        self._current_audio_stream = _AudioStream()
        self._current_text_stream = _TextStream()
        self._full_transcript = ""

        modalities_fut: asyncio.Future[list[Literal["text", "audio"]]] = (
            asyncio.get_event_loop().create_future()
        )
        modalities_fut.set_result(["audio", "text"])

        generation = MessageGeneration(
            message_id=f"vllm-msg-{int(time.time())}",
            text_stream=self._current_text_stream,
            audio_stream=self._current_audio_stream,
            modalities=modalities_fut,
        )

        event = GenerationCreatedEvent(
            message_stream=_MessageStream(generation),
            function_stream=_EmptyFunctionStream(),
            user_initiated=False,
        )

        if self._generation_future and not self._generation_future.done():
            self._generation_future.set_result(event)
            self._generation_future = None

        self.emit("generation_created", event)

    def _finish_generation(self) -> None:
        logger.info("Finishing generation — closing streams")
        if self._current_text_stream:
            self._current_text_stream.close()
            self._current_text_stream = None
        if self._current_audio_stream:
            self._current_audio_stream.close()
            self._current_audio_stream = None

    async def aclose(self) -> None:
        self._closed = True
        self._finish_generation()
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._model._sessions.discard(self)
