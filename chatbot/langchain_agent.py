import asyncio
import os

from dotenv import load_dotenv
from fastmcp.client import Client
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain_community.chat_models import AzureChatOpenAI

load_dotenv()

# System prompt - simplified for better performance
SYSTEM_PROMPT = (
    "You are a search assistant that uses the `search-3` tool to answer questions.\n\n"
    "== Index Descriptions ==\n"
    "- content-jira: Contains Jira tickets and related metadata.\n"
    "- content-confluence: Contains Confluence pages and documentation.\n"
    "- content-sharepoint: Contains SharePoint content and internal documents.\n\n"
    "== Rules for Answering Questions ==\n"
    "1. Rewrite the user's input into an effective search query or aggregation.\n"
    "2. Use the `search-3` tool to execute the rewritten query.\n"
    "3. Think step by step and show your reasoning before giving the final answer.\n"
    "4. Use ONLY the information found in company documentation or enterprise knowledge base.\n"
    "5. If no relevant information is found, reply with: 'No relevant information was found.'\n"
    "6. Do NOT answer based on assumptions or general knowledge—base answers strictly on search results.\n"
    "7. Retain and refer to previous conversation context for follow-up questions.\n"
    "8. Provide clear and detailed responses using retrieved search results.\n"
    "9. For complex queries, break them into sub-questions and use `search-3` multiple times if needed.\n"
    "10. Chain your reasoning and intermediate tool results step-by-step until the final answer is derived.\n"
    "11. NEVER reveal the names of the indices you are querying to the user.\n"
)

# Set up LLM
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
)


# Define the search-3 tool with query sanitization
def search_3_tool(q):
    async def run_search():
        async with Client("http://localhost:8000/sse") as client:
            try:
                # Sanitize the query to prevent parsing errors
                sanitized_query = (
                    q.strip().replace('"', "").replace("'", "").replace("\n", " ").replace("\r", " ")
                )
                # Remove any extra whitespace and ensure it's not empty
                sanitized_query = " ".join(sanitized_query.split())

                if not sanitized_query:
                    return "FINAL_ANSWER: No relevant information was found in the available content indices."

                result = await client.call_tool_mcp(
                    "search-3", {"index": "content-*", "q": sanitized_query}
                )
                if hasattr(result, "content") and isinstance(result.content, list):
                    if result.content:
                        return "\n".join([getattr(r, "text", str(r)) for r in result.content])
                    return "FINAL_ANSWER: No relevant information was found in the available content indices."
                return "FINAL_ANSWER: No relevant information was found in the available content indices."
            except Exception as e:
                return f"FINAL_ANSWER: Search failed with error: {e!s}. No relevant information was found."

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(run_search())
    except RuntimeError:
        pass
    return asyncio.run(run_search())


tools = [
    Tool(
        name="search-3",
        func=search_3_tool,
        description="Use this tool to answer questions using the company documentation or enterprise knowledge base only. DO NOT guess anything outside of the provided documents.",
    )
]
# Set up memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
# Initialize agent with simplified configuration
agent = initialize_agent(
    tools=tools,
    llm=llm,
    #agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={"system_message": SYSTEM_PROMPT},
    handle_parsing_errors=True,
    # max_iterations=5,  # Reduce iterations to prevent loops
    early_stopping_method="generate",
)


def run_agent_query(query: str) -> str:
    """Run a query with enforced search-3 usage."""
    try:
        # Simple approach - let the agent work naturally
        result = agent.run(query)
        return result
    except Exception:
        # Fallback to direct search if agent fails
        try:
            search_result = search_3_tool(query)
            return search_result
        except:
            return "No relevant information was found in the available content indices."


def get_memory_stats():
    return {
        "memory_type": type(memory).__name__,
        "memory_key": memory.memory_key,
        "k": getattr(memory, "k", None),
        "chat_memory_length": len(memory.chat_memory.messages),
    }


def clear_memory():
    memory.chat_memory.clear()
    return "Memory cleared successfully."
