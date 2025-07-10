import re
from typing import Any
import logging

import chainlit as cl
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import asyncio
from enhanced_agent_manager import LangGraphElasticsearchAgent, AgentConfig

load_dotenv()

agent = None  # Will be set in on_chat_start

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class EnhancedChainlitCallbackHandler(BaseCallbackHandler):
    """Enhanced callback handler with selective step display"""
    def __init__(self):
        self.steps = []
        self.current_step = None
        self.step_count = 0
        self.reasoning_steps = []
        self.search_results = []
        self.show_thinking = True
        self.show_search = True
        self.show_final_reasoning = True

async def run_agent_with_selective_steps(query: str, show_thinking=True, show_search=True, show_final=True):
    callback_handler = EnhancedChainlitCallbackHandler()
    callback_handler.show_thinking = show_thinking
    callback_handler.show_search = show_search
    callback_handler.show_final_reasoning = show_final
    try:
        # You may need to adapt this to your agent's API
        result = await agent.process_query(query)
        return result
    except Exception as e:
        async with cl.Step(name="⚠️ Fallback Search") as error_step:
            await error_step.stream_token(f"Primary agent failed: {e!s}\n")
            await error_step.stream_token("🔄 Trying direct search...\n")
            await error_step.stream_token(f"❌ All methods failed: {e!s}\n")
            return "No relevant information was found in the available content indices."

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def on_chat_start():
    global agent
    logger.info("Chat session started. Initializing agent.")
    config = AgentConfig(
        max_steps=int(os.getenv("AGENT_MAX_STEPS", 10)),
        temperature=float(os.getenv("AGENT_TEMPERATURE", 0.1)),
        enable_console_logging=True
    )
    agent = LangGraphElasticsearchAgent(config)
    logger.debug(f"Agent config: {config}")
    #Connect to MCP server
    server_command = "uv"
    server_args = ["run", "elasticsearch-mcp-server"]
    es_env = {
        "ELASTICSEARCH_HOSTS": os.getenv("ELASTICSEARCH_HOSTS", "https://localhost:9200"),
        "ELASTICSEARCH_USERNAME": os.getenv("ELASTICSEARCH_USERNAME", "elastic"),
        "ELASTICSEARCH_PASSWORD": os.getenv("ELASTICSEARCH_PASSWORD", "test123"),
        **os.environ,
    }
    logger.info("Connecting to MCP server...")
    await agent.connect_to_mcp_server(server_command, server_args, env=es_env)
    logger.info("Connected to MCP server.")
    
    elements = [
        cl.Text(content="What are the projects lead by Dayananda", name="text1"),
        cl.Text(content="List the success criteria for the Department of Defence project", name="text2"),
        cl.Text(content="Go live updates of the westpac project", name="text3"),
        cl.Text(content="Compare the success criteria of the Department of Defence project with those of the Westpac project", name="text4"),
    ]

    # Setting elements will open the sidebar
    await cl.ElementSidebar.set_elements(elements)
    await cl.ElementSidebar.set_title("Example Questions")

    await cl.Message(
        content=(
            "🤖 **Project Portfolio Assistant**\n\n"
            "Ask questions about your enterprise content. I'll show you my reasoning process "
            "with collapsible steps that you can expand to see details.\n\n"
            "**Features:**\n"
            "- RAG backed by Elasticsearch\n"
            "- Multi Step Searches\n"
            "- Multi Hop Reasoning\n"
            "- Guardrails for the search results\n"
            "- Context-aware conversations"
        ),
        author="assistant",
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    global agent
    logger.info(f"Received message: {message.content}")
    user_id = cl.user_session.get("user_id")
    if not user_id:
        import uuid
        user_id = f"chainlit_{uuid.uuid4().hex[:8]}"
        cl.user_session.set("user_id", user_id)
    session_id = cl.user_session.get("session_id")
    if not session_id:
        import uuid
        session_id = f"chainlit_{uuid.uuid4().hex[:8]}"
        cl.user_session.set("session_id", session_id)
    logger.debug(f"User ID: {user_id}, Session ID: {session_id}")
    
    # Call the agent and show dynamic sub-steps for transparency
    async with cl.Step(name="Checking the Portfolio") as trace_step:
        logger.info("Processing query with agent...")
        result = await agent.process_query(message.content, user_id, session_id)
        logger.debug(f"Agent result: {result}")

        # Show queries as sub-steps if present
        queries = result.get("queries_executed", [])
        if queries:
            for i, q in enumerate(queries, 1):
                logger.info(f"Query {i}: {q['query']}")
                async with cl.Step(name=f"Query {i}", type="query") as qstep:
                    qstep.input = q['query']['purpose']
                    qstep.output = f"{q['query']['query_type']}\n```json\n{q['query']['query_body']}\n```"
        # Show reasoning steps as sub-steps if present
        reasoning = result.get("reasoning_steps", [])
        if reasoning:
            for step in reasoning:
                logger.info(f"Reasoning Step {step['step_number']}: {step['description']} | Action: {step['action']}")
                async with cl.Step(name=f"Reasoning Step {step['step_number']}", type="reasoning") as rstep:
                    rstep.input = step['description']
                    rstep.output = f"Action: {step['action']}"
    if "error" in result:
        logger.error(f"Error in agent result: {result['error']}")
        await cl.Message(content=f"❌ Error: {result['error']}" ).send()
        return
    # Main answer at the bottom
    answer = result.get('answer') or "No answer generated."
    logger.info(f"Final answer: {answer}")
    await cl.Message(content=answer).send()
    

@cl.on_chat_end
async def on_chat_end():
    global agent
    if agent:
        await agent.cleanup()
