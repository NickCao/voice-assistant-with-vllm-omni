from __future__ import annotations

import asyncio
import base64
import io
import json as json_mod
import logging
import time
import wave
from collections.abc import AsyncIterator
from typing import Literal
from urllib.parse import urlparse

from livekit import rtc
from livekit.agents.utils.aio import Chan
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.realtime import (
    GenerationCreatedEvent,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeSession,
)
from livekit.agents.llm import FunctionCall
from livekit.agents.llm.tool_context import Tool, ToolChoice, ToolContext
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

logger = logging.getLogger("vllm-realtime")

MAX_TOOL_ROUNDS = 5


class _ToolCallInfo:
    __slots__ = ("call_id", "name", "arguments")

    def __init__(self, *, call_id: str, name: str, arguments: str) -> None:
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class _CompletionResult:
    __slots__ = ("text", "tool_calls")

    def __init__(self, *, text: str, tool_calls: list[_ToolCallInfo]) -> None:
        self.text = text
        self.tool_calls = tool_calls

    def to_assistant_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return msg



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



def _frames_to_wav_base64(frames: list[rtc.AudioFrame]) -> str:
    if not frames:
        return ""
    pcm = b"".join(bytes(f.data) for f in frames)
    sample_rate = frames[0].sample_rate
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def _wav_bytes_to_frame(wav_data: bytes) -> rtc.AudioFrame | None:
    buf = io.BytesIO(wav_data)
    try:
        with wave.open(buf, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            pcm = w.readframes(w.getnframes())
    except Exception:
        return None
    samples_per_channel = len(pcm) // (2 * ch)
    if samples_per_channel == 0:
        return None
    return rtc.AudioFrame(
        data=pcm, sample_rate=sr, num_channels=ch,
        samples_per_channel=samples_per_channel,
    )


class VLLMRealtimeModel(RealtimeModel):
    def __init__(
        self, *, base_url: str, model: str, api_key: str = "EMPTY",
        speaker: str = "Ethan",
        room: rtc.Room | None = None,
    ) -> None:
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=False,
                manual_function_calls=False,
            )
        )
        self._base_url = base_url
        self._model_name = model
        self._api_key = api_key
        self._speaker = speaker
        self._room = room
        self._sessions: set[VLLMRealtimeSession] = set()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return urlparse(self._base_url).hostname or "vllm-omni"

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
        self._client = AsyncOpenAI(
            api_key=realtime_model._api_key, base_url=realtime_model._base_url,
        )
        self._closed = False
        self._chat_ctx = ChatContext()
        self._tool_ctx = ToolContext([])
        self._instructions = ""
        self._conversation: list[dict] = []
        self._audio_buffer: list[rtc.AudioFrame] = []
        self._current_audio_stream: _AudioStream | None = None
        self._current_text_stream: _TextStream | None = None
        self._generation_task: asyncio.Task | None = None
        self._interrupted = False
        self._audio_source: rtc.AudioSource | None = None
        self._audio_track_published = False

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
        self._audio_buffer.append(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.get_event_loop().create_future()

        frames = list(self._audio_buffer)
        self._audio_buffer.clear()

        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

        tc = tool_choice if isinstance(tool_choice, str) else "auto"
        self._generation_task = asyncio.create_task(
            self._run_generation(frames, fut, tool_choice=tc)
        )
        return fut

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        self._audio_buffer.clear()

    def interrupt(self) -> None:
        logger.info("Interrupt requested")
        self._interrupted = True
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        self._finish_generation()

    def truncate(
        self, *, message_id: str, modalities: list[Literal["text", "audio"]],
        audio_end_ms: int, audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    async def _run_generation(
        self,
        frames: list[rtc.AudioFrame],
        fut: asyncio.Future[GenerationCreatedEvent],
        tool_choice: str = "auto",
    ) -> None:
        wav_b64 = _frames_to_wav_base64(frames)
        if not wav_b64:
            logger.warning("No audio to process")
            if not fut.done():
                fut.set_exception(RuntimeError("No audio to process"))
            return

        logger.info("Generating reply from %d audio frames", len(frames))

        messages: list[dict] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        messages.extend(self._conversation)
        user_message = {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"}},
            ],
        }
        messages.append(user_message)

        self._current_text_stream = _TextStream()
        self._current_function_stream: Chan[FunctionCall] = Chan()

        modalities_fut: asyncio.Future[list[Literal["text", "audio"]]] = (
            asyncio.get_event_loop().create_future()
        )
        modalities_fut.set_result(["text"])

        generation = MessageGeneration(
            message_id=f"vllm-msg-{int(time.time())}",
            text_stream=self._current_text_stream,
            audio_stream=_AudioStream(),
            modalities=modalities_fut,
        )

        event = GenerationCreatedEvent(
            message_stream=_MessageStream(generation),
            function_stream=self._current_function_stream,
            user_initiated=False,
        )

        if not fut.done():
            fut.set_result(event)
        self.emit("generation_created", event)

        tools_param = None
        if tool_choice != "none":
            tools_param = self._tool_ctx.parse_function_tools("openai") or None
        logger.info("Tools available: %d (tool_choice=%s)", len(tools_param) if tools_param else 0, tool_choice)

        generation_start = time.perf_counter()

        try:
            result = await self._streaming_completion(
                messages, tools_param, tool_choice, generation_start,
            )

            new_history: list[dict] = [user_message]

            for _round in range(MAX_TOOL_ROUNDS):
                if not result.tool_calls:
                    break

                logger.info("Executing %d tool calls (round %d)", len(result.tool_calls), _round + 1)
                assistant_msg = result.to_assistant_message()
                messages.append(assistant_msg)
                new_history.append(assistant_msg)

                tool_results = await self._execute_tools(result.tool_calls)
                messages.extend(tool_results)
                new_history.extend(tool_results)

                result = await self._streaming_completion(
                    messages, tools_param, "auto", generation_start,
                )

            if result.text:
                new_history.append({"role": "assistant", "content": result.text})
            self._conversation.extend(new_history)

        except asyncio.CancelledError:
            logger.info("Generation cancelled")
        except Exception:
            logger.exception("Chat completion error")

        self._finish_generation()

    async def _streaming_completion(
        self,
        messages: list[dict],
        tools_param: list | None,
        tool_choice: str,
        generation_start: float,
    ) -> _CompletionResult:
        kwargs: dict = {
            "model": self._model._model_name,
            "modalities": ["text", "audio"],
            "messages": messages,
            "stream": True,
            "extra_body": {"speaker": self._model._speaker},
        }
        if tools_param and tool_choice != "none":
            kwargs["tools"] = tools_param
            kwargs["tool_choice"] = tool_choice

        state = ChatCompletionStreamState(input_tools=tools_param or [])
        stream = await self._client.chat.completions.create(**kwargs)

        text = ""
        tool_calls: list[_ToolCallInfo] = []
        first_audio = True

        async for chunk in stream:
            for ev in state.handle_chunk(chunk):
                if ev.type == "tool_calls.function.arguments.done":
                    snap_tc = (state.current_completion_snapshot.choices[0].message.tool_calls or [])[ev.index]
                    call_id = getattr(snap_tc, "id", "") or f"call_{ev.index}"
                    logger.info("Tool call: %s(%s) [%s]", ev.name, ev.arguments[:100], call_id)
                    tool_calls.append(_ToolCallInfo(call_id=call_id, name=ev.name, arguments=ev.arguments))
                    if self._model._room:
                        await self._model._room.local_participant.publish_data(
                            json_mod.dumps({"name": ev.name, "arguments": ev.arguments}).encode(),
                            topic="tool_call",
                        )

            modality = getattr(chunk, "modality", None)
            for choice in chunk.choices:
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue
                content = getattr(delta, "content", None)

                if modality == "audio" and content:
                    if first_audio:
                        first_audio = False
                        ttfa = time.perf_counter() - generation_start
                        logger.info("Time to first audio: %.3fs", ttfa)
                        interrupted = self._interrupted
                        self._interrupted = False
                        if self._model._room:
                            await self._model._room.local_participant.publish_data(
                                f'{{"ttfa": {ttfa:.3f}, "interrupted": {str(interrupted).lower()}}}'.encode(),
                                topic="latency",
                            )
                    frame = _wav_bytes_to_frame(base64.b64decode(content))
                    if frame:
                        await self._ensure_audio_source()
                        await self._audio_source.capture_frame(frame)
                elif content:
                    text += content
                    if self._current_text_stream:
                        self._current_text_stream.push(content)

        return _CompletionResult(text=text, tool_calls=tool_calls)

    async def _execute_tools(self, tool_calls: list[_ToolCallInfo]) -> list[dict]:
        from livekit.agents.llm.utils import execute_function_call
        from livekit.agents.llm.llm import FunctionToolCall

        results = []
        for tc in tool_calls:
            ftc = FunctionToolCall(call_id=tc.call_id, name=tc.name, arguments=tc.arguments)
            result = await execute_function_call(ftc, self._tool_ctx)
            content = result.fnc_call_out.output if result.fnc_call_out else ""
            if result.raw_exception:
                content = f"Error: {result.raw_exception}"
            logger.info("Tool result for %s: %s", tc.name, str(content)[:200])
            results.append({
                "role": "tool",
                "tool_call_id": tc.call_id,
                "content": str(content) if content else "",
            })
        return results

    async def _ensure_audio_source(self) -> None:
        if self._audio_source is None:
            self._audio_source = rtc.AudioSource(24000, 1)
        if not self._audio_track_published and self._model._room:
            track = rtc.LocalAudioTrack.create_audio_track("assistant-audio", self._audio_source)
            await self._model._room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            self._audio_track_published = True

    def _finish_generation(self) -> None:
        if self._current_text_stream:
            self._current_text_stream.close()
            self._current_text_stream = None
        if getattr(self, "_current_function_stream", None) is not None:
            self._current_function_stream.close()
            self._current_function_stream = None

    async def aclose(self) -> None:
        self._closed = True
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            try:
                await self._generation_task
            except (asyncio.CancelledError, Exception):
                pass
        self._finish_generation()
        await self._client.close()
        self._model._sessions.discard(self)
