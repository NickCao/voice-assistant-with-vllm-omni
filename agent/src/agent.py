import logging
import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe, JobContext, cli
from livekit.agents.llm import function_tool
from livekit.plugins import silero

from vllm_realtime import VLLMRealtimeModel

load_dotenv(".env.local")
logger = logging.getLogger("voice-assistant")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8091/v1")

server = AgentServer()


@function_tool()
async def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72°F."


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Respond naturally and concisely. You have access to tools — use them when appropriate.",
            tools=[get_weather],
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
    )
    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )

    logger.info("Voice assistant started, connected to vLLM-Omni at %s", VLLM_BASE_URL)


if __name__ == "__main__":
    cli.run_app(server)
