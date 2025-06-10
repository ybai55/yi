from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, ModelSettings
import os
from dotenv import load_dotenv

load_dotenv("./.env")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

def main():
    external_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    model = OpenAIChatCompletionsModel( 
        model="qwq:32b",
        openai_client=external_client,
    )
    # Create the agent
    agent = Agent(
        name="Assistant",
        instructions="Replay in one sentence",
        model=model,
        model_settings=ModelSettings(temperature=0.5),
    )

    # Run the agent synchronously
    result = Runner.run_sync(agent, "What is the capital of Brazil?")
    print(result.final_output)


if __name__ == "__main__":
    main()
