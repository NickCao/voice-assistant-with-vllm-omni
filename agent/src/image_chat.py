"""Add image uploads from the frontend to the active Realtime conversation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass, field

from livekit import rtc
from livekit.agents import AgentSession
from livekit.agents.llm import ImageContent

logger = logging.getLogger("image-chat")

IMAGE_CHAT_TOPIC = "image_chat"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_CHUNK_CHARS = 10_000
MAX_CHUNKS = 800
MAX_ACTIVE_UPLOADS_PER_PARTICIPANT = 2
UPLOAD_TIMEOUT_SECONDS = 60


@dataclass
class _ImageUpload:
    participant_identity: str
    request_id: str
    mime_type: str
    expected_size: int
    total_chunks: int
    chunks: dict[int, str] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.monotonic)


class ImageChatHandler:
    """Receive chunked image uploads and add them to the active realtime session."""

    def __init__(
        self,
        *,
        room: rtc.Room,
        realtime_session: AgentSession,
        realtime_ready: asyncio.Event | None = None,
    ) -> None:
        self._room = room
        self._realtime_session = realtime_session
        self._realtime_ready = realtime_ready
        self._uploads: dict[tuple[str, str], _ImageUpload] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

        self._room.on("data_received", self._on_data_received)
        self._room.on("disconnected", self._on_disconnected)

    def _on_disconnected(self, *_args) -> None:
        if not self._closed:
            asyncio.create_task(self.aclose())

    def _on_data_received(self, packet: rtc.DataPacket) -> None:
        if self._closed or packet.topic != IMAGE_CHAT_TOPIC or packet.participant is None:
            return

        try:
            message = json.loads(packet.data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict):
            return

        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 64:
            return

        self._expire_uploads()
        participant_identity = packet.participant.identity
        message_type = message.get("type")
        if message_type == "image_start":
            self._start_upload(participant_identity, request_id, message)
        elif message_type == "image_chunk":
            self._append_chunk(participant_identity, request_id, message)

    def _start_upload(
        self, participant_identity: str, request_id: str, message: dict
    ) -> None:
        mime_type = message.get("mime_type")
        expected_size = message.get("size")
        total_chunks = message.get("total_chunks")

        error = None
        if not isinstance(mime_type, str) or mime_type not in {
            "image/jpeg", "image/png", "image/webp"
        }:
            error = "Choose a JPEG, PNG, or WebP image."
        elif type(expected_size) is not int or not 0 < expected_size <= MAX_IMAGE_BYTES:
            error = "Images must be 5 MB or smaller."
        elif type(total_chunks) is not int or not 0 < total_chunks <= MAX_CHUNKS:
            error = "The image upload could not be read. Please try again."
        elif total_chunks != (
            ((expected_size + 2) // 3 * 4 + MAX_CHUNK_CHARS - 1) // MAX_CHUNK_CHARS
        ):
            error = "The image upload could not be read. Please try again."

        if error:
            self._send_error(participant_identity, request_id, error)
            return

        active_uploads = sum(
            1 for upload in self._uploads.values()
            if upload.participant_identity == participant_identity
        )
        if active_uploads >= MAX_ACTIVE_UPLOADS_PER_PARTICIPANT:
            self._send_error(
                participant_identity,
                request_id,
                "Another image is still uploading. Please try again shortly.",
            )
            return

        self._uploads[(participant_identity, request_id)] = _ImageUpload(
            participant_identity=participant_identity,
            request_id=request_id,
            mime_type=mime_type,
            expected_size=expected_size,
            total_chunks=total_chunks,
        )

    def _append_chunk(
        self, participant_identity: str, request_id: str, message: dict
    ) -> None:
        key = (participant_identity, request_id)
        upload = self._uploads.get(key)
        if upload is None:
            return

        index = message.get("index")
        chunk = message.get("data")
        if (
            type(index) is not int
            or not 0 <= index < upload.total_chunks
            or not isinstance(chunk, str)
            or not 0 < len(chunk) <= MAX_CHUNK_CHARS
        ):
            self._uploads.pop(key, None)
            self._send_error(
                participant_identity,
                request_id,
                "The image upload was incomplete. Please try again.",
            )
            return

        encoded_size = (upload.expected_size + 2) // 3 * 4
        expected_chunk_size = min(
            MAX_CHUNK_CHARS, encoded_size - index * MAX_CHUNK_CHARS
        )
        if len(chunk) != expected_chunk_size:
            self._uploads.pop(key, None)
            self._send_error(
                participant_identity,
                request_id,
                "The image upload was incomplete. Please try again.",
            )
            return

        upload.chunks[index] = chunk
        upload.updated_at = time.monotonic()
        if len(upload.chunks) != upload.total_chunks:
            return

        self._uploads.pop(key, None)
        image_base64 = "".join(upload.chunks[i] for i in range(upload.total_chunks))
        self._spawn(self._add_image(upload, image_base64))

    def _expire_uploads(self) -> None:
        now = time.monotonic()
        expired = [
            key for key, upload in self._uploads.items()
            if now - upload.updated_at > UPLOAD_TIMEOUT_SECONDS
        ]
        for key in expired:
            self._uploads.pop(key, None)

    def _send_error(self, participant_identity: str, request_id: str, error: str) -> None:
        self._spawn(
            self._send_message(
                {
                    "type": "image_error",
                    "request_id": request_id,
                    "error": error,
                },
                participant_identity,
            )
        )

    async def _add_image(self, upload: _ImageUpload, image_base64: str) -> None:
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError):
            await self._send_error_message(upload, "The image upload could not be read.")
            return

        if len(image_bytes) != upload.expected_size or not self._matches_mime_type(
            image_bytes, upload.mime_type
        ):
            await self._send_error_message(upload, "The selected file is not a valid image.")
            return

        await self._send_message(
            {"type": "image_status", "request_id": upload.request_id},
            upload.participant_identity,
        )

        try:
            await self._add_image_to_realtime_context(upload, image_base64)
        except Exception:
            logger.exception(
                "Adding image to Realtime context failed for request %s", upload.request_id
            )
            await self._send_error_message(
                upload,
                "I couldn't add that image to the conversation. Please try again.",
            )
            return

        await self._send_message(
            {
                "type": "image_ready",
                "request_id": upload.request_id,
                "text": "Image added. Ask about it by voice.",
            },
            upload.participant_identity,
        )

    async def _add_image_to_realtime_context(
        self, upload: _ImageUpload, image_base64: str
    ) -> None:
        if self._realtime_ready is not None:
            await self._realtime_ready.wait()

        image_url = f"data:{upload.mime_type};base64,{image_base64}"
        chat_ctx = self._realtime_session.current_agent.chat_ctx.copy()
        chat_ctx.add_message(
            role="user",
            content=[ImageContent(image=image_url)],
        )
        await self._realtime_session.current_agent.update_chat_ctx(chat_ctx)

    async def _send_error_message(self, upload: _ImageUpload, error: str) -> None:
        await self._send_message(
            {
                "type": "image_error",
                "request_id": upload.request_id,
                "error": error,
            },
            upload.participant_identity,
        )

    async def _send_message(self, message: dict, participant_identity: str) -> None:
        if self._closed:
            return
        await self._room.local_participant.publish_data(
            json.dumps(message, ensure_ascii=False).encode("utf-8"),
            reliable=True,
            destination_identities=[participant_identity],
            topic=IMAGE_CHAT_TOPIC,
        )

    @staticmethod
    def _matches_mime_type(image_bytes: bytes, mime_type: str) -> bool:
        if mime_type == "image/jpeg":
            return image_bytes.startswith(b"\xff\xd8\xff")
        if mime_type == "image/png":
            return image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        if mime_type == "image/webp":
            return image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"
        return False

    def _spawn(self, coroutine) -> None:
        if self._closed:
            coroutine.close()
            return
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._room.off("data_received", self._on_data_received)
        self._room.off("disconnected", self._on_disconnected)
        self._uploads.clear()

        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
