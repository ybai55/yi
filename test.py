from app.agents.agent_factory import AgentFactory

import asyncio

async def main():
    agent_factory = AgentFactory()
    agent_executor = agent_factory.create_agent_executor(1, 1)
    result = agent_factory.stream_agent_response(agent_executor, "给我讲一个笑话",[], 1, 1, 1)
    async for response in result:
        print(response)
    # for response in agent_factory.stream_agent_response(agent_executor, "给我讲一个笑话",[], 1, 1, 1):
        # print(response)

if __name__ == '__main__':
    
    asyncio.run(main())
