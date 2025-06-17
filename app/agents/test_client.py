from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv("./.env")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


def create_ollama_client(model="qwq:32b", temperature=0.5, max_retries=3):


    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    # msg = llm.invoke(f"Write a joke about time")
    # print({"joke": msg.content})    

    return llm


