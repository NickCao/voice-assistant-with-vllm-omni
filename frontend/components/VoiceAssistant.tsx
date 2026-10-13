"use client";

import {
  LiveKitRoom,
  RoomAudioRenderer,
  BarVisualizer,
  DisconnectButton,
  useRoomContext,
  useVoiceAssistant,
  useDataChannel,
  useTranscriptions,
} from "@livekit/components-react";
import "@livekit/components-styles";
import Image from "next/image";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from "react";

interface ConnectionDetails {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
}

type AgentMode = "completions" | "realtime";

const AGENT_MODES: { value: AgentMode; label: string }[] = [
  { value: "completions", label: "Completions API" },
  { value: "realtime", label: "Realtime API" },
];

interface LatencyPoint {
  ms: number;
  interrupted: boolean;
  toolCall: boolean;
}

interface ToolCallEntry {
  name: string;
  arguments: string;
  timestamp: number;
}

const MAX_LATENCY_POINTS = 20;
const IMAGE_CHAT_TOPIC = "image_chat";
const IMAGE_CHUNK_SIZE = 10_000;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const ACCEPTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

interface ImageTurn {
  id: string;
  imageUrl: string;
  fileName: string;
  status: "sending" | "adding" | "ready" | "error";
  response?: string;
}

interface SelectedImage {
  file: File;
  previewUrl: string;
}

interface ImageChatEvent {
  type?: string;
  request_id?: string;
  text?: string;
  error?: string;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("Could not read the selected image."));
    };
    reader.onerror = () => reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(file);
  });
}

function ImageChatPanel() {
  const room = useRoomContext();
  const [turns, setTurns] = useState<ImageTurn[]>([]);
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const previewUrlsRef = useRef(new Set<string>());

  useDataChannel(IMAGE_CHAT_TOPIC, (message) => {
    try {
      const event: ImageChatEvent = JSON.parse(
        new TextDecoder().decode(message.payload),
      );
      if (!event.request_id) return;

      if (event.type === "image_status") {
        setTurns((previous) =>
          previous.map((turn) =>
            turn.id === event.request_id ? { ...turn, status: "adding" } : turn,
          ),
        );
      } else if (event.type === "image_ready") {
        setTurns((previous) =>
          previous.map((turn) =>
            turn.id === event.request_id
              ? {
                  ...turn,
                  status: "ready",
                  response: event.text || "Image added. Ask about it by voice.",
                }
              : turn,
          ),
        );
      } else if (event.type === "image_error") {
        setTurns((previous) =>
          previous.map((turn) =>
            turn.id === event.request_id
              ? { ...turn, status: "error", response: event.error || "Could not add image." }
              : turn,
          ),
        );
      }
    } catch {
      // Ignore unrelated or malformed data-channel messages.
    }
  });

  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [turns]);

  useEffect(() => {
    return () => {
      for (const url of previewUrlsRef.current) URL.revokeObjectURL(url);
    };
  }, []);

  const handleImageSelected = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;

    if (!ACCEPTED_IMAGE_TYPES.has(file.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError("Images must be 5 MB or smaller.");
      return;
    }

    if (selectedImage) {
      URL.revokeObjectURL(selectedImage.previewUrl);
      previewUrlsRef.current.delete(selectedImage.previewUrl);
    }
    const previewUrl = URL.createObjectURL(file);
    previewUrlsRef.current.add(previewUrl);
    setSelectedImage({ file, previewUrl });
    setError(null);
  };

  const handleRemoveImage = () => {
    if (selectedImage) {
      URL.revokeObjectURL(selectedImage.previewUrl);
      previewUrlsRef.current.delete(selectedImage.previewUrl);
    }
    setSelectedImage(null);
  };

  const handleSend = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedImage || sending) return;

    const requestId = crypto.randomUUID();
    const image = selectedImage;
    const turn: ImageTurn = {
      id: requestId,
      imageUrl: image.previewUrl,
      fileName: image.file.name,
      status: "sending",
    };
    setTurns((previous) => [...previous, turn]);
    setSelectedImage(null);
    setSending(true);
    setError(null);

    try {
      const dataUrl = await readFileAsDataUrl(image.file);
      const base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
      const totalChunks = Math.ceil(base64.length / IMAGE_CHUNK_SIZE);
      const publish = (payload: Record<string, string | number>) =>
        room.localParticipant.publishData(
          new TextEncoder().encode(JSON.stringify(payload)),
          { reliable: true, topic: IMAGE_CHAT_TOPIC },
        );

      await publish({
        type: "image_start",
        request_id: requestId,
        mime_type: image.file.type,
        size: image.file.size,
        total_chunks: totalChunks,
      });

      // LiveKit data packets have a size limit, so split the base64 payload
      // into reliable packets and let the agent reassemble it.
      for (let index = 0; index < totalChunks; index += 1) {
        await publish({
          type: "image_chunk",
          request_id: requestId,
          index,
          data: base64.slice(index * IMAGE_CHUNK_SIZE, (index + 1) * IMAGE_CHUNK_SIZE),
        });
      }
    } catch (sendError) {
      const message =
        sendError instanceof Error ? sendError.message : "Could not send the image.";
      setTurns((previous) =>
        previous.map((current) =>
          current.id === requestId
            ? { ...current, status: "error", response: message }
            : current,
        ),
      );
      setError(message);
    } finally {
      setSending(false);
    }
  };

  return (
    <section className="w-full max-w-lg rounded-xl border border-zinc-800 bg-zinc-950 p-4">
      <div className="mb-3">
        <h2 className="text-sm font-semibold text-zinc-100">Ask about an image</h2>
        <p className="mt-1 text-xs text-zinc-500">
          Send a JPEG, PNG, or WebP image up to 5 MB, then ask about it by voice.
        </p>
      </div>

      {turns.length > 0 && (
        <div ref={scrollRef} className="mb-4 flex max-h-80 flex-col gap-3 overflow-y-auto">
          {turns.map((turn) => (
            <div key={turn.id} className="rounded-lg bg-zinc-900 p-3">
              <div className="flex gap-3">
                <Image
                  src={turn.imageUrl}
                  alt={turn.fileName}
                  width={112}
                  height={84}
                  unoptimized
                  className="h-20 w-28 rounded-md object-cover"
                />
                <div className="min-w-0">
                  <p className="text-xs text-sky-300">Image</p>
                  <p className="mt-1 break-words text-sm text-zinc-200">{turn.fileName}</p>
                </div>
              </div>
              <div className="mt-3 border-t border-zinc-800 pt-3">
                <p className="text-xs text-emerald-300">Status</p>
                <p className="mt-1 whitespace-pre-wrap break-words text-sm text-zinc-300">
                  {turn.response ||
                    (turn.status === "adding"
                      ? "Adding image to the conversation…"
                      : turn.status === "sending"
                        ? "Sending image…"
                        : "")}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      <form onSubmit={handleSend} className="flex flex-col gap-3">
        {selectedImage && (
          <div className="flex items-center gap-3 rounded-lg bg-zinc-900 p-2">
            <Image
              src={selectedImage.previewUrl}
              alt={selectedImage.file.name}
              width={80}
              height={60}
              unoptimized
              className="h-14 w-20 rounded object-cover"
            />
            <p className="min-w-0 flex-1 truncate text-sm text-zinc-300">
              {selectedImage.file.name}
            </p>
            <button
              type="button"
              onClick={handleRemoveImage}
              className="px-2 text-sm text-zinc-400 hover:text-white"
              aria-label="Remove selected image"
            >
              Remove
            </button>
          </div>
        )}

        <div className="flex items-center justify-between gap-3">
          <label className="cursor-pointer rounded-full border border-zinc-700 px-4 py-2 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white">
            Choose image
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleImageSelected}
              className="sr-only"
            />
          </label>
          <button
            type="submit"
            disabled={!selectedImage || sending}
            className="rounded-full bg-white px-5 py-2 text-sm font-medium text-black transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {sending ? "Sending…" : "Send image"}
          </button>
        </div>
      </form>

      {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
    </section>
  );
}

function LatencyChart({ points }: { points: LatencyPoint[] }) {
  if (points.length === 0) return null;

  const ms = points.map((p) => p.ms);
  const max = Math.max(...ms, 100);
  const h = 160;
  const w = 480;
  const barW = w / MAX_LATENCY_POINTS;
  const normalPoints = points.filter((p) => !p.interrupted && !p.toolCall);
  const normalMs = normalPoints.map((p) => p.ms);

  return (
    <div className="w-full max-w-lg">
      <div
        className="flex items-end justify-between rounded-lg bg-zinc-900 p-4"
        style={{ height: h + 32 }}
      >
        <svg
          width={w}
          height={h}
          viewBox={`0 0 ${w} ${h}`}
          className="w-full"
        >
          {points.map((p, i) => {
            const barH = (p.ms / max) * (h - 20);
            const x = i * barW;
            const color = p.toolCall
              ? "#38bdf8"
              : p.interrupted
                ? "#a78bfa"
                : p.ms < 300
                  ? "#22c55e"
                  : p.ms < 500
                    ? "#eab308"
                    : "#ef4444";
            return (
              <g key={i}>
                <rect
                  x={x + 2}
                  y={h - barH}
                  width={barW - 4}
                  height={barH}
                  rx={3}
                  fill={color}
                  opacity={p.interrupted || p.toolCall ? 0.5 : 0.85}
                />
                {p.interrupted && (
                  <line
                    x1={x + 2}
                    y1={h - barH + 4}
                    x2={x + barW - 4}
                    y2={h - 4}
                    stroke={color}
                    strokeWidth={1}
                    opacity={0.6}
                  />
                )}
                <text
                  x={x + barW / 2}
                  y={h - barH - 4}
                  textAnchor="middle"
                  fontSize={10}
                  fill="#a1a1aa"
                >
                  {Math.round(p.ms)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-zinc-500">
        <span>
          Last:{" "}
          <span className="text-zinc-300">
            {Math.round(ms[ms.length - 1])}ms
          </span>
        </span>
        <span>
          Avg:{" "}
          <span className="text-zinc-300">
            {normalMs.length > 0
              ? Math.round(
                  normalMs.reduce((a, b) => a + b, 0) / normalMs.length,
                )
              : "—"}
            ms
          </span>
        </span>
        <span>
          Min:{" "}
          <span className="text-zinc-300">
            {normalMs.length > 0 ? Math.round(Math.min(...normalMs)) : "—"}ms
          </span>
        </span>
        <span>
          <span className="text-sky-400">■</span> tool
          {" "}
          <span className="text-violet-400">■</span> interrupted
        </span>
      </div>
    </div>
  );
}

function ToolCallLog({ entries }: { entries: ToolCallEntry[] }) {
  if (entries.length === 0) return null;

  return (
    <div className="w-full max-w-lg">
      <div className="rounded-lg bg-zinc-900 p-3">
        <p className="mb-2 text-xs font-medium text-zinc-400">Tool Calls</p>
        <div className="flex flex-col gap-1.5 max-h-40 overflow-y-auto">
          {entries.map((entry, i) => (
            <div key={i} className="rounded bg-zinc-800 px-2.5 py-1.5 text-xs">
              <span className="font-mono text-emerald-400">{entry.name}</span>
              <span className="text-zinc-500 ml-1 break-all">
                ({entry.arguments.length > 80
                  ? entry.arguments.slice(0, 80) + "…"
                  : entry.arguments})
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TranscriptPanel() {
  // Only the assistant's speech is transcribed — neither agent backend
  // transcribes user audio — so every segment here is the assistant's.
  const transcriptions = useTranscriptions();
  const scrollRef = useRef<HTMLDivElement>(null);

  const sorted = [...transcriptions].sort(
    (a, b) => a.streamInfo.timestamp - b.streamInfo.timestamp,
  );
  const lastText = sorted[sorted.length - 1]?.text;

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [sorted.length, lastText]);

  if (sorted.length === 0) return null;

  return (
    <div className="w-full max-w-lg">
      <div
        ref={scrollRef}
        className="flex max-h-56 flex-col gap-2 overflow-y-auto rounded-lg bg-zinc-900 p-4"
      >
        {sorted.map((t) => (
          <p key={t.streamInfo.id} className="text-sm">
            <span className="text-sky-400">Assistant:</span>{" "}
            <span className="text-zinc-200">{t.text}</span>
          </p>
        ))}
      </div>
    </div>
  );
}

function AgentVisualizer() {
  const { state, audioTrack } = useVoiceAssistant();
  const [points, setPoints] = useState<LatencyPoint[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCallEntry[]>([]);

  useDataChannel("latency", (msg) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(msg.payload));
      if (typeof data.ttfa === "number") {
        setPoints((prev) => [
          ...prev.slice(-MAX_LATENCY_POINTS + 1),
          { ms: data.ttfa * 1000, interrupted: data.interrupted === true, toolCall: data.tool_call === true },
        ]);
      }
    } catch {}
  });

  useDataChannel("tool_call", (msg) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(msg.payload));
      if (typeof data.name === "string") {
        setToolCalls((prev) => [
          ...prev.slice(-9),
          { name: data.name, arguments: data.arguments || "", timestamp: Date.now() },
        ]);
      }
    } catch {}
  });

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="h-48 w-full max-w-md">
        <BarVisualizer state={state} barCount={5} trackRef={audioTrack} />
      </div>
      <p className="text-sm text-zinc-400 capitalize">{state}</p>
      <TranscriptPanel />
      <ToolCallLog entries={toolCalls} />
      <LatencyChart points={points} />
    </div>
  );
}

export default function VoiceAssistant() {
  const [connectionDetails, setConnectionDetails] =
    useState<ConnectionDetails | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [agentMode, setAgentMode] = useState<AgentMode>("completions");

  const handleConnect = useCallback(async () => {
    setConnecting(true);
    try {
      const response = await fetch("/api/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agentMode }),
      });
      if (!response.ok) throw new Error("Failed to get token");
      const details: ConnectionDetails = await response.json();
      setConnectionDetails(details);
    } catch (err) {
      console.error("Connection failed:", err);
      setConnecting(false);
    }
  }, [agentMode]);

  const handleDisconnected = useCallback(() => {
    setConnectionDetails(null);
    setConnecting(false);
  }, []);

  if (!connectionDetails) {
    return (
      <div className="flex flex-col items-center gap-6">
        <div className="flex rounded-full bg-zinc-900 p-1">
          {AGENT_MODES.map((mode) => (
            <button
              key={mode.value}
              onClick={() => setAgentMode(mode.value)}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                agentMode === mode.value
                  ? "bg-white text-black"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-zinc-500">
          Image input is available in Realtime API mode.
        </p>
        <button
          onClick={handleConnect}
          disabled={connecting}
          className="rounded-full bg-white px-8 py-4 text-lg font-medium text-black transition-opacity hover:opacity-80 disabled:opacity-50"
        >
          {connecting ? "Connecting..." : "Start Conversation"}
        </button>
      </div>
    );
  }

  return (
    <LiveKitRoom
      token={connectionDetails.participantToken}
      serverUrl={connectionDetails.serverUrl}
      connect={true}
      audio={true}
      onDisconnected={handleDisconnected}
      className="flex flex-col items-center gap-8"
    >
      {agentMode === "realtime" && <ImageChatPanel />}
      <AgentVisualizer />
      <RoomAudioRenderer />
      <DisconnectButton className="rounded-full border border-zinc-700 px-6 py-3 text-sm text-zinc-300 transition-colors hover:border-red-500 hover:text-red-400">
        End Conversation
      </DisconnectButton>
    </LiveKitRoom>
  );
}
