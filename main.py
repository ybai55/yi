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
        instructions="你是一个精通周易的道家师傅。",
        model=model,
        model_settings=ModelSettings(temperature=0.5),
    )

    # Run the agent synchronously
    result = Runner.run_sync(agent, "用梅花易数的时间卜卦法，帮我算一下今天的宜忌")
    print(result.final_output)


if __name__ == "__main__":
    main()
