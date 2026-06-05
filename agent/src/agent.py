import logging
import os

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, cli
from livekit.plugins import openai

load_dotenv(".env.local")
logger = logging.getLogger("voice-assistant")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8091/v1")


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    model = openai.realtime.RealtimeModel(
        base_url=VLLM_BASE_URL,
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        api_key="not-needed",
        modalities=["audio", "text"],
    )

    agent = openai.realtime.MultimodalAgent(model=model)
    agent.start(ctx.room)

    logger.info("Voice assistant started, connected to vLLM-Omni at %s", VLLM_BASE_URL)


if __name__ == "__main__":
    cli.run_app(entrypoint)
