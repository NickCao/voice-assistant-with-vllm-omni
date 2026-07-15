import logging
import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe, JobContext, cli, mcp
from livekit.plugins import silero

from vllm_realtime import VLLMRealtimeModel

load_dotenv(".env.local")
logger = logging.getLogger("voice-assistant")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8091/v1")

server = AgentServer()


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Respond naturally and concisely. You have access to tools — use them when appropriate.",
        )


@server.rtc_session(agent_name="voice-assistant")
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    model = VLLMRealtimeModel(
        base_url=VLLM_BASE_URL,
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        room=ctx.room,
    )

    session = AgentSession(
        llm=model,
        vad=silero.VAD.load(),
        turn_detection="vad",
        mcp_servers=[mcp.MCPServerHTTP(url="https://instances-mcp.vantage.sh/mcp/c7d0bef0-0b04-4141-a2dc-1b0f850b2c29")],
    )
    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )

    logger.info("Voice assistant started, connected to vLLM-Omni at %s", VLLM_BASE_URL)


if __name__ == "__main__":
    cli.run_app(server)
