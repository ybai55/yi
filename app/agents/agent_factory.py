from typing import List, Dict, Any, Tuple
from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from app.llm.llm_factory import LLMFactory, BaseLLM
from app.utils.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)

class AgentFactory:
    def __init__(self):
        self.llm_provider = LLMFactory.get_llm(settings.DEFAULT_LLM_PROVIDER)

    def create_agent_executor(self, user_id: int, conversation_id: int) -> CompiledGraph:
        """ 
        Creates and returns a LangGraph CompiledGraph instance for a specific user.
        """
        llm = self.llm_provider.get_model()
        agent_executor = create_react_agent(model=llm, tools=[])

        logger.info(f"LangGraph CompiledGraph instance created for user: {user_id}, conversation: {conversation_id}")
        return agent_executor

    async def stream_agent_response(
            self, 
            agent_executor: CompiledGraph, 
            user_input: str, 
            chat_history: List[BaseMessage],
            user_id: int,
            conversation_id: int,
            user_message_id: int,
        ) -> AsyncIterator[dict[str, Any] | Any]:
        """
        Streams responses from the LangGraph agent executor.
        """
        # callback_handler
        inputs = {
            "input": user_input,
        }

        full_response_content = ""

        try:
            logger.info("starting agent streaming")
            async for chunk in agent_executor.astream(inputs, stream_mode="update"):
                print(chunk)

        except Exception as e:
            logger.error(f"Error during agent streaming for user {user_id}: {e}", exc_info=True)
            yield f"An error occurred: {str(e)}"
        
        finally:
            # After streaming is complete (or errors), save token usage and final response
            # token_summary = callback_handler.get_current_token_summary()
            # await self.chat_history_service.add_message_to_history(
            #     conversation_id=conversation_id, 
            #     role="assistant", 
            #     content=full_response_content, 
            #     token_usage=token_summary.get("completion_tokens", 0) # Store completion tokens for assistant
            # )
            # Update user's input message with prompt tokens (if not done previously)
            # This requires updating an existing message, which our current add_message_to_history doesn't do.
            # A more robust solution would be to update the message in the DB after LLM response for prompt tokens.
            # For simplicity, we assume token_usage passed to add_message_to_history only tracks current message's output.
            logger.info(f"Agent stream finished. Final response length: {len(full_response_content)}. Token summary: ")

