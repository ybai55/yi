import asyncio
import random

from agents import (
    Agent, Runner, function_tool,
    OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_export_api_key
)
from openai.types.responses import ResponseTextDeltaEvent

import os
from dotenv import load_dotenv
import mlflow


load_dotenv("./.env")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MLFLOW_URL = os.getenv("MLFLOW_URL")

set_tracing_export_api_key(os.getenv(API_KEY))

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("Ollama")

@function_tool
def how_many_jokes() -> int:
    return random.randint(1, 10)


async def main():
    external_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    model = OpenAIChatCompletionsModel( 
        model="qwq:32b",
        openai_client=external_client,
    )

    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
        model=model,
        tools=[how_many_jokes],
    )

    result = Runner.run_streamed(agent, input="Please tell me 5 jokes.")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

    # === Run starting ===
    # Agent updated: Joker
    # -- Tool was called
    # -- Tool output: 4
    # -- Message output:
    #  Sure, here are four jokes for you:

    # 1. **Why don't skeletons fight each other?**
    #    They don't have the guts!

    # 2. **What do you call fake spaghetti?**
    #    An impasta!

    # 3. **Why did the scarecrow win an award?**
    #    Because he was outstanding in his field!

    # 4. **Why did the bicycle fall over?**
    #    Because it was two-tired!
    # === Run complete ===