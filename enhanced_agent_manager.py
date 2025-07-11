"""
Enhanced LangGraph Elasticsearch Agent with Intelligent Query Building
Provides sophisticated conversation management, memory persistence, and intelligent routing
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableConfig

# Additional imports
from dotenv import load_dotenv
import sqlite3
from pydantic import BaseModel, Field

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# State Management
class AgentState(TypedDict):
    """The state of the agent conversation"""
    messages: List[AnyMessage]
    query: str
    user_id: str
    session_id: str
    elasticsearch_queries: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    query_analysis: Dict[str, Any]
    step_count: int
    max_steps: int
    reasoning_steps: List[str]
    final_answer: str
    metadata: Dict[str, Any]
    search_needed: bool
    conversational_response: str
    contextualized_query: str

# Pydantic models for structured data
class QueryAnalysis(BaseModel):
    """Analysis of user query intent and requirements"""
    intent: str = Field(description="The intent behind the query")
    complexity: Literal["simple", "moderate", "complex"] = Field(description="Complexity level")
    requires_aggregation: bool = Field(description="Whether query needs aggregation")
    requires_multi_step: bool = Field(description="Whether query needs multiple steps")
    elasticsearch_strategy: str = Field(description="Strategy for elasticsearch querying")
    confidence: float = Field(description="Confidence in the analysis")
    key_concepts: List[str] = Field(description="Key concepts extracted from query")

class ElasticsearchQuery(BaseModel):
    """Structured Elasticsearch query"""
    query_type: str = Field(description="Type of query (search, aggregation, etc.)")
    query_body: Dict[str, Any] = Field(description="The actual ES query DSL")
    purpose: str = Field(description="Purpose of this query")
    expected_result_type: str = Field(description="Expected type of results")

class ReasoningStep(BaseModel):
    """A single reasoning step in multi-step analysis"""
    step_number: int
    description: str
    action: str
    query_needed: bool
    elasticsearch_query: Optional[ElasticsearchQuery] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class AgentConfig:
    """Configuration for the agent"""
    max_steps: int = 10
    temperature: float = 0.1
    max_tokens: int = 2000
    enable_console_logging: bool = True
    elasticsearch_timeout: int = 30
    similarity_threshold: float = 0.7

class ElasticsearchQueryBuilder:
    """Intelligent Elasticsearch query builder"""
    
    def __init__(self, llm: AzureChatOpenAI, get_mappings_func, get_sample_func=None):
        self.llm = llm
        self.get_mappings = get_mappings_func
        self.get_sample = get_sample_func
        
    @staticmethod
    def extract_json_from_llm_response(text: str) -> str:
        import re
        text = text.strip()
        if text.startswith('```'):
            text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
            text = re.sub(r'```$', '', text)
        return text.strip()

    @staticmethod
    def extract_field_list(mapping: dict) -> str:
        import json
        # Ensure mapping is a dict
        if isinstance(mapping, str):
            try:
                mapping = json.loads(mapping)
            except Exception:
                return "(No fields found in mapping)"
        # Try to extract properties from typical ES mapping structure
        properties = None
        if 'mappings' in mapping and 'properties' in mapping['mappings']:
            properties = mapping['mappings']['properties']
        elif 'properties' in mapping:
            properties = mapping['properties']
        else:
            # fallback: try to find first properties dict
            for v in mapping.values():
                if isinstance(v, dict) and 'properties' in v:
                    properties = v['properties']
                    break
        if not properties:
            return "(No fields found in mapping)"
        return '\n'.join([f"- {k} ({v.get('type', 'unknown')})" for k, v in properties.items()])

    @staticmethod
    def get_field_set(mapping: dict) -> set:
        import json
        # Ensure mapping is a dict
        if isinstance(mapping, str):
            try:
                mapping = json.loads(mapping)
            except Exception:
                return set()
        # Returns a set of field names for validation
        properties = None
        if 'mappings' in mapping and 'properties' in mapping['mappings']:
            properties = mapping['mappings']['properties']
        elif 'properties' in mapping:
            properties = mapping['properties']
        else:
            for v in mapping.values():
                if isinstance(v, dict) and 'properties' in v:
                    properties = v['properties']
                    break
        if not properties:
            return set()
        return set(properties.keys())

    async def build_search_query(self, query: str, analysis: QueryAnalysis, index_name: str) -> ElasticsearchQuery:
        """Build an Elasticsearch search query based on analysis and index mapping"""
        logger.info(f"Building search query for: {query} on index: {index_name}")
        index_mapping = await self.get_mappings(index_name)
        logger.info(f"Full mapping for '{index_name}':\n{json.dumps(index_mapping, indent=2)}")
        sample_doc = None
        if self.get_sample:
            sample_doc = await self.get_sample(index_name)
            if sample_doc:
                logger.info(f"Sample document from '{index_name}':\n{json.dumps(sample_doc, indent=2)}")
        prompt = f"""
You are an expert Elasticsearch query builder.

User Query: {query}
Query Analysis: {analysis.model_dump()}

The full mapping for the '{index_name}' index is:
{json.dumps(index_mapping, indent=2)}
"""
        if sample_doc:
            prompt += f"\nA sample document from this index is:\n{json.dumps(sample_doc, indent=2)}\n"
        prompt += """
Instructions:
- For filtering (including in aggregation queries), use "text" fields with "match" or "match_phrase" queries to allow partial matches, unless the user requests an exact match.
- Use ".keyword" fields only for aggregation buckets (e.g., "terms", "value_count") or when an exact match is explicitly required.
- Do NOT use "term" queries on ".keyword" fields for filtering unless the user query is an exact value.
- Use ONLY the fields and structure shown in the mapping above.
- You may use nested, keyword, or text fields as appropriate.
- If you are unsure which field to use, explain your reasoning in a comment.
- If the mapping is ambiguous, you may suggest a query using a wildcard or ask for clarification.

Return ONLY a valid JSON object with the following fields:
- query_type: (search, aggregation, complex_search, etc.)
- query_body: (the actual ES query DSL)
- purpose: (what this query is trying to achieve)
- expected_result_type: (documents, aggregations, etc.)

For example:
{
  "query_type": "search",
  "query_body": { ... },
  "purpose": "...",
  "expected_result_type": "documents"
}
"""
        messages = [SystemMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        try:
            json_str = self.extract_json_from_llm_response(response.content)
            query_data = json.loads(json_str)
            return ElasticsearchQuery(**query_data)
        except Exception as e:
            logger.error(f"Error parsing query response: {e} | LLM response: {response.content}")
            return ElasticsearchQuery(
                query_type="search",
                query_body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["*"],
                            "type": "best_fields"
                        }
                    },
                    "highlight": {"fields": {"*": {}}},
                    "size": 10
                },
                purpose="Simple text search",
                expected_result_type="documents"
            )

    async def build_aggregation_query(self, query: str, analysis: QueryAnalysis, index_name: str) -> ElasticsearchQuery:
        """Build an Elasticsearch aggregation query using index mapping"""
        logger.info(f"Building aggregation query for: {query} on index: {index_name}")
        index_mapping = await self.get_mappings(index_name)
        logger.info(f"Full mapping for '{index_name}':\n{json.dumps(index_mapping, indent=2)}")
        sample_doc = None
        if self.get_sample:
            sample_doc = await self.get_sample(index_name)
            if sample_doc:
                logger.info(f"Sample document from '{index_name}':\n{json.dumps(sample_doc, indent=2)}")
        prompt = f"""
You are an expert Elasticsearch query builder.

User Query: {query}
Query Analysis: {analysis.model_dump()}

The full mapping for the '{index_name}' index is:
{json.dumps(index_mapping, indent=2)}
"""
        if sample_doc:
            prompt += f"\nA sample document from this index is:\n{json.dumps(sample_doc, indent=2)}\n"
        prompt += """
Instructions:
- For filtering (including in aggregation queries), use "text" fields with "match" or "match_phrase" queries to allow partial matches, unless the user requests an exact match.
- Use ".keyword" fields only for aggregation buckets (e.g., "terms", "value_count") or when an exact match is explicitly required.
- Do NOT use "term" queries on ".keyword" fields for filtering unless the user query is an exact value.
- Use ONLY the fields and structure shown in the mapping above.
- You may use nested, keyword, or text fields as appropriate.
- If you are unsure which field to use, explain your reasoning in a comment.
- If the mapping is ambiguous, you may suggest a query using a wildcard or ask for clarification.

Return ONLY a valid JSON object with the following fields:
- query_type: (search, aggregation, complex_search, etc.)
- query_body: (the actual ES query DSL)
- purpose: (what this query is trying to achieve)
- expected_result_type: (documents, aggregations, etc.)

For example:
{
  "query_type": "aggregation",
  "query_body": { ... },
  "purpose": "...",
  "expected_result_type": "aggregations"
}
"""
        messages = [SystemMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        try:
            json_str = self.extract_json_from_llm_response(response.content)
            query_data = json.loads(json_str)
            return ElasticsearchQuery(**query_data)
        except Exception as e:
            logger.error(f"Error parsing aggregation response: {e} | LLM response: {response.content}")
            return ElasticsearchQuery(
                query_type="aggregation",
                query_body={
                    "size": 0,
                    "aggs": {
                        "results": {
                            "terms": {
                                "field": "_type",
                                "size": 10
                            }
                        }
                    }
                },
                purpose="Basic aggregation",
                expected_result_type="aggregations"
            )

class LangGraphElasticsearchAgent:
    """Enhanced Elasticsearch agent using LangGraph"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Initialize query builder, pass in the get_mappings function
        self.query_builder = ElasticsearchQueryBuilder(self.llm, self._get_mappings, self._get_sample_doc)
        
        # Initialize checkpointer for memory - Use MemorySaver instead of SqliteSaver
        self.checkpointer = MemorySaver()
        
        # MCP session will be set by connect method
        self.mcp_session = None
        
        # Build the graph
        self.graph = self._build_graph()
        
        self.index_mapping_cache = {}  # index_name -> mapping
        
        logger.info("LangGraph Elasticsearch Agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("plan_steps", self._plan_steps_node)
        workflow.add_node("execute_step", self._execute_step_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("final_response", self._final_response_node)
        
        # Add edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "plan_steps")
        workflow.add_edge("plan_steps", "execute_step")
        workflow.add_conditional_edges(
            "execute_step",
            self._should_continue_execution,
            {
                "continue": "execute_step",
                "synthesize": "synthesize_answer"
            }
        )
        workflow.add_edge("synthesize_answer", "final_response")
        workflow.add_edge("final_response", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    # MCP Tool functions (to be called directly, not as LangChain tools)
    async def _elasticsearch_search(self, index: str, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Elasticsearch search query via MCP server"""
        if not self.mcp_session:
            return {"error": "MCP session not initialized"}
        
        try:
            logger.info(f"Executing ES search on index: {index}")
            logger.info(f"Query body: {json.dumps(query_body, indent=2)}")
            
            # Switch back to the search_documents tool
            result = await self.mcp_session.call_tool(
                "search_documents",
                {"index": index, "body": query_body}
            )
            
            return result.content[0].text if result.content else {"error": "No response from MCP"}
            
        except Exception as e:
            logger.error(f"Elasticsearch search error: {e}")
            return {"error": str(e)}
    
    async def _get_mappings(self, index: str) -> Dict[str, Any]:
        if index in self.index_mapping_cache:
            return self.index_mapping_cache[index]
        if not self.mcp_session:
            return {"error": "MCP session not initialized"}
        try:
            logger.info(f"Getting mappings for index: {index}")
            result = await self.mcp_session.call_tool(
                "get_index",
                {"index": index}
            )
            mapping = result.content[0].text if result.content else '{"error": "No response from MCP"}'
            logger.info(f"Full mapping response: {mapping}")
            if isinstance(mapping, str):
                try:
                    mapping = json.loads(mapping)
                except Exception:
                    logger.error("Failed to parse mapping JSON")
                    return {}
            self.index_mapping_cache[index] = mapping  # Cache it!
            return mapping
        except Exception as e:
            logger.error(f"Get mappings error: {e}")
            return {"error": str(e)}
    
    async def _get_sample_doc(self, index: str) -> Optional[Dict[str, Any]]:
        if not self.mcp_session:
            return None
        try:
            logger.info(f"Getting sample document for index: {index}")
            # Switch back to the search_documents tool
            result = await self.mcp_session.call_tool(
                "search_documents",
                {"index": index, "body": {"size": 1, "query": {"match_all": {}}}}
            )
            if result.content and hasattr(result.content[0], "text"):
                try:
                    doc = json.loads(result.content[0].text)
                    if "hits" in doc and "hits" in doc["hits"] and doc["hits"]["hits"]:
                        return doc["hits"]["hits"][0]
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.error(f"Get sample doc error: {e}")
            return None

    async def _build_contextual_step_query(self, current_step: dict, original_query: str, analysis: QueryAnalysis, state: AgentState, step_count: int) -> str:
        """Build a context-aware query for the current step using results from previous steps"""
        
        # If this is the first step, use the original query
        if step_count == 0:
            return original_query
        
        # Get previous search results
        previous_results = state.get("search_results", [])
        if not previous_results:
            return original_query
        
        # Build context from previous steps
        context_prompt = f"""
        Build a specific search query for this step based on previous results:
        
        Original Query: {original_query}
        Current Step: {current_step['description']}
        Step Number: {step_count + 1}
        
        Previous Search Results:
        {json.dumps(previous_results, indent=2)}
        
        Instructions:
        - Use information from previous steps to make this query more specific and targeted
        - Extract key terms, entity names, IDs, or other identifiers from previous results
        - Build upon the context established in previous steps
        - Make the query more focused based on what was discovered
        
        **General Approach:**
        - If previous step found entities: Use those entity names/IDs in the current query
        - If previous step found categories: Filter by those categories
        - If previous step found time periods: Use those time constraints
        - If previous step found properties: Search for related properties or details
        - Always make the query more specific than the original, using context from previous steps
        
        Return ONLY the search query string, nothing else.
        """
        
        try:
            messages = [SystemMessage(content=context_prompt)]
            response = await self.llm.ainvoke(messages)
            contextual_query = response.content.strip()
            
            # Clean up the response
            if contextual_query.startswith('"') and contextual_query.endswith('"'):
                contextual_query = contextual_query[1:-1]
            
            logger.info(f"Built contextual query for step {step_count + 1}: {contextual_query}")
            return contextual_query
            
        except Exception as e:
            logger.error(f"Error building contextual query: {e}")
            return original_query

    # Node implementations
    async def _analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze the user query to understand intent and requirements"""
        import re
        query = state["query"]
        # Use the full conversation history for better context/pronoun resolution
        history = "\n".join([
            m.content for m in state["messages"] if hasattr(m, "content")
        ])

        logger.info(f"Analyzing query: {query}")
        analysis_prompt = f"""
Conversation history:
{history}

Current user query: {query}

Instructions:
- Rewrite the current user query as a fully-contextualized, standalone query, incorporating all relevant context from the conversation. Output this as 'contextualized_query'.

Analyze this Elasticsearch query request:

User Query: {query}

**MULTI-STEP ANALYSIS GUIDELINES:**
Set requires_multi_step to TRUE if the query involves ANY of these characteristics:

**Pattern-Based Detection:**
- "summarize from all projects" or "summarize across projects"
- "what we have done" or "work done" or "accomplishments" 
- "compare" or "comparison" between different entities
- "find all" + "then" + another action
- "list" + "and" + "analyze" or "summarize"

**General Complexity Detection:**
- Queries requiring data gathering followed by analysis/synthesis
- Queries with multiple distinct information needs or sub-questions
- Queries that need to find entities first, then analyze their properties
- Queries involving aggregation across multiple entities or categories
- Queries requiring sequential reasoning (find X, then analyze Y based on X)
- Queries with temporal or hierarchical relationships (timeline, phases, categories)
- Queries needing to combine information from different sources or perspectives
- Queries requiring both search and analysis phases

**Decision Framework:**
Ask yourself: "Can this query be answered with a single search, or does it need multiple steps?"
- Single search: Direct lookup, simple filtering, basic aggregation
- Multiple steps: Find entities → analyze properties → synthesize results

**COMPLEXITY GUIDELINES:**
- simple: Single search term, direct lookup
- moderate: Multiple search terms, filtering, basic aggregation
- complex: Multi-step reasoning, data gathering + analysis, comparisons, summarization

Determine:
1. intent: What is the user trying to find/analyze?
2. complexity: simple, moderate, or complex
3. requires_aggregation: Does this need aggregations/analytics?
4. requires_multi_step: Does this need multiple searches? (Use guidelines above)
5. elasticsearch_strategy: What ES approach is best?
6. confidence: How confident are you in this analysis?
7. key_concepts: Extract the main concepts
8. search_needed: true if this is a substantive search/analytics request, false if this is a greeting/chit-chat/irrelevant
9. conversational_response: (string) Friendly reply if search_needed is false, otherwise empty or null
10. contextualized_query: (string) The user query rewritten as a fully-contextualized, standalone query, with all pronouns and references resolved.

Return ONLY a valid JSON object with these fields.
"""
        messages = [SystemMessage(content=analysis_prompt)]
        response = await self.llm.ainvoke(messages)
        def _parse_confidence(conf):
            if isinstance(conf, float) or isinstance(conf, int):
                return float(conf)
            if isinstance(conf, str):
                mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
                return mapping.get(conf.lower(), 0.5)
            return 0.5
        try:
            json_str = ElasticsearchQueryBuilder.extract_json_from_llm_response(response.content)
            analysis_data = json.loads(json_str)
            if "confidence" in analysis_data:
                analysis_data["confidence"] = _parse_confidence(analysis_data["confidence"])
            # Patch: If search_needed is false, ensure elasticsearch_strategy is a string
            if not analysis_data.get("search_needed", True):
                if not analysis_data.get("elasticsearch_strategy"):
                    analysis_data["elasticsearch_strategy"] = "none"
            query_analysis = QueryAnalysis(**{k: v for k, v in analysis_data.items() if k in QueryAnalysis.model_fields})
            logger.info(f"Query analysis complete: {query_analysis.model_dump()}")
            state["query_analysis"] = query_analysis.model_dump()
            state["step_count"] = 0
            state["metadata"] = {"analysis_timestamp": datetime.now().isoformat()}
            # Handle search_needed and conversational_response
            state["search_needed"] = analysis_data.get("search_needed", True)
            state["conversational_response"] = analysis_data.get("conversational_response", "")
            state["contextualized_query"] = analysis_data.get("contextualized_query", query)
        except Exception as e:
            logger.error(f"Error analyzing query: {e} | LLM response: {response.content}")
            state["query_analysis"] = {
                "intent": "search",
                "complexity": "moderate",
                "requires_aggregation": False,
                "requires_multi_step": False,
                "elasticsearch_strategy": "simple_search",
                "confidence": 0.5,
                "key_concepts": [query]
            }
            state["search_needed"] = True
            state["conversational_response"] = ""
            state["contextualized_query"] = query
        return state
    
    async def _plan_steps_node(self, state: AgentState) -> AgentState:
        """Plan the execution steps based on query analysis"""
        # If search is not needed, skip planning
        if not state.get("search_needed", True):
            state["reasoning_steps"] = []
            return state
        query = state.get("contextualized_query", state["query"])
        analysis = state["query_analysis"]
        
        logger.info(f"Planning execution steps for query: {query}")
        
        if analysis.get("requires_multi_step", False):
            # Create multi-step plan
            planning_prompt = f"""
            Create a step-by-step execution plan for this complex query:
            
            Query: {query}
            Analysis: {json.dumps(analysis, indent=2)}
            
            **GENERAL MULTI-STEP PLANNING PRINCIPLES:**
            
            **Step 1: Data Discovery/Gathering**
            - Find relevant entities, projects, or categories
            - Identify the scope of data needed
            - Use broad search to understand what's available
            
            **Step 2: Data Extraction/Analysis**
            - Extract specific properties, metrics, or details from discovered entities
            - Filter, aggregate, or analyze the gathered data
            - Focus on the specific information requested
            
            **Step 3: Synthesis/Comparison**
            - Combine results from previous steps
            - Compare, summarize, or synthesize findings
            - Provide the final answer or insights
            
            **Adaptive Planning:**
            - Analyze the query structure and identify natural breakpoints
            - Consider what information needs to be gathered first
            - Plan steps that build upon each other logically
            - Ensure each step has a clear, specific purpose
            
            **Common Patterns:**
            - Entity Discovery → Property Analysis → Synthesis
            - Data Gathering → Aggregation → Summary
            - Comparison Setup → Data Collection → Comparison Analysis
            - Timeline Creation → Event Analysis → Trend Identification
            
            Break this down into logical steps that can be executed sequentially.
            Each step should be a specific Elasticsearch operation.
            
            Return ONLY a valid JSON array of steps, each with:
            - step_number: int
            - description: str (be specific about what this step will search for)
            - action: str (search, aggregate, analyze, etc.)
            - query_needed: bool
            """
            
            messages = [SystemMessage(content=planning_prompt)]
            response = await self.llm.ainvoke(messages)
            
            try:
                # Extract JSON from markdown code blocks if present
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.startswith('```'):
                    content = content[3:]  # Remove ```
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                
                steps_data = json.loads(content.strip())
                reasoning_steps = [ReasoningStep(**step) for step in steps_data]
                state["reasoning_steps"] = [step.model_dump() for step in reasoning_steps]
                
                logger.info(f"Planned {len(reasoning_steps)} execution steps")
                
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error planning steps: {e} | LLM response: {response.content}")
                # Fallback to single step
                state["reasoning_steps"] = [{
                    "step_number": 1,
                    "description": "Execute single search query",
                    "action": "search",
                    "query_needed": True
                }]
        else:
            # Simple single-step plan
            state["reasoning_steps"] = [{
                "step_number": 1,
                "description": "Execute search query",
                "action": "search" if not analysis.get("requires_aggregation") else "aggregate",
                "query_needed": True
            }]
        
        return state
    
    async def _execute_step_node(self, state: AgentState) -> AgentState:
        """Execute the current reasoning step"""
        # If search is not needed, skip execution
        if not state.get("search_needed", True):
            return state
        step_count = state["step_count"]
        reasoning_steps = state["reasoning_steps"]
        
        if step_count >= len(reasoning_steps):
            logger.info("All steps completed")
            return state
        
        current_step = reasoning_steps[step_count]
        logger.info(f"Executing step {step_count + 1}: {current_step['description']}")
        
        if current_step["query_needed"]:
            # Build and execute query with context from previous steps
            query = state.get("contextualized_query", state["query"])
            analysis = QueryAnalysis(**state["query_analysis"])
            
            # Build context-aware query based on previous results
            step_query = await self._build_contextual_step_query(
                current_step, query, analysis, state, step_count
            )
            
            # Use the correct index for project portfolio context
            import os
            index_name = os.getenv("ELASTICSEARCH_INDEX_NAME", "text_project_portfolio")
            if current_step["action"] == "aggregate":
                es_query = await self.query_builder.build_aggregation_query(step_query, analysis, index_name)
            else:
                es_query = await self.query_builder.build_search_query(step_query, analysis, index_name)
            
            # Execute the query
            search_result = await self._elasticsearch_search(index_name, es_query.query_body)
            
            # Store the results
            if "elasticsearch_queries" not in state:
                state["elasticsearch_queries"] = []
            if "search_results" not in state:
                state["search_results"] = []
            
            state["elasticsearch_queries"].append({
                "step": step_count + 1,
                "query": es_query.model_dump(),
                "executed": True
            })
            
            state["search_results"].append({
                "step": step_count + 1,
                "result": search_result,
                "query_type": es_query.query_type
            })
            
            logger.info(f"Executed query for step {step_count + 1}: {es_query.query_type}")
        
        # Update step count
        state["step_count"] = step_count + 1
        
        return state
    
    def _should_continue_execution(self, state: AgentState) -> str:
        """Determine whether to continue execution or synthesize"""
        step_count = state["step_count"]
        reasoning_steps = state["reasoning_steps"]
        max_steps = state.get("max_steps", self.config.max_steps)
        
        if step_count >= len(reasoning_steps):
            logger.info("All reasoning steps completed - moving to synthesis")
            return "synthesize"
        
        if step_count >= max_steps:
            logger.info("Maximum steps reached - moving to synthesis")
            return "synthesize"
        
        logger.info("Continuing with next step")
        return "continue"
    
    async def _synthesize_answer_node(self, state: AgentState) -> AgentState:
        """Synthesize the final answer from all search results"""
        # If search is not needed, just return the conversational response
        if not state.get("search_needed", True):
            state["final_answer"] = state.get("conversational_response", "I'm here to help!")
            return state
        query = state["query"]
        search_results = state.get("search_results", [])
        reasoning_steps = state["reasoning_steps"]
        
        logger.info(f"Synthesizing answer for query: {query}")
        
        synthesis_prompt = f"""
        Synthesize a comprehensive answer based on the multi-step search results:
        
        Original Query: {query}
        
        Execution Steps: {json.dumps(reasoning_steps, indent=2)}
        
        Search Results: {json.dumps(search_results, indent=2)}
        
        **MULTI-STEP SYNTHESIS GUIDELINES:**
        
        For "summarize from all projects" queries:
        - Step 1 typically finds projects matching criteria
        - Step 2 extracts work/accomplishments from those projects
        - Step 3 synthesizes the findings into a comprehensive summary
        
        For "what we have done" queries:
        - Step 1 finds relevant entities/projects
        - Step 2 extracts accomplishments and work done
        - Step 3 summarizes the work and achievements
        
        For comparison queries:
        - Step 1 gathers data for first entity
        - Step 2 gathers data for second entity
        - Step 3 compares and analyzes differences
        
        **SYNTHESIS INSTRUCTIONS:**
        1. Follow the logical flow of the execution steps
        2. Use results from each step to build a coherent narrative
        3. For project summaries: List projects found, then summarize work done across those projects
        4. For comparisons: Present data for each entity, then highlight key differences
        5. For work summaries: Identify entities/projects, then summarize accomplishments
        6. Acknowledge if any steps returned limited or no results
        7. Suggest follow-up questions based on the findings
        
        Provide a clear, comprehensive answer that directly addresses the user's query.
        Be conversational but informative, and structure the response logically.
        """
        
        messages = [SystemMessage(content=synthesis_prompt)]
        response = await self.llm.ainvoke(messages)
        
        state["final_answer"] = response.content
        
        logger.info("Answer synthesis complete")
        
        return state
    
    async def _final_response_node(self, state: AgentState) -> AgentState:
        """Generate the final response with metadata"""
        final_answer = state["final_answer"]
        
        # Add AI message to conversation
        ai_message = AIMessage(content=final_answer)
        state["messages"] = add_messages(state["messages"], [ai_message])
        
        # Update metadata
        state["metadata"].update({
            "completion_timestamp": datetime.now().isoformat(),
            "total_steps": state["step_count"],
            "queries_executed": len(state.get("elasticsearch_queries", [])),
            "reasoning_steps_count": len(state["reasoning_steps"])
        })
        
        logger.info("Final response generated")
        
        return state
    
    async def connect_to_mcp_server(self, server_command: str = "uv", server_args: list = None, env: dict = None, cwd: str = None):
        """Connect to MCP server via stdio (for uv run elasticsearch-mcp-server)"""
        try:
            import sys
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.client.session import ClientSession
            
            logger.info(f"Connecting to MCP server via stdio: {server_command} {server_args or []}")
            params = StdioServerParameters(
                command=server_command,
                args=server_args or [],
                env=env,
                cwd=cwd
            )
            
            # Start the MCP server process and get the read/write streams
            self._stdio_context = stdio_client(params)
            read_stream, write_stream = await self._stdio_context.__aenter__()
            
            # Create a session
            self._session_context = ClientSession(read_stream, write_stream)
            self.mcp_session = await self._session_context.__aenter__()
            await self.mcp_session.initialize()
            
            # Optionally, list tools to confirm connection
            try:
                response = await self.mcp_session.list_tools()
                tools = response.tools
                logger.info(f"Connected to MCP server. Available tools: {[tool.name for tool in tools]}")
            except Exception as e:
                logger.warning(f"Connected to MCP server but could not list tools: {e}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.mcp_session = None
    
    def _get_message_history_from_memory(self, session_id: str) -> list:
        """Retrieve the full message history for a session/thread from the checkpointer."""
        config = RunnableConfig(configurable={"thread_id": session_id})
        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)
            if checkpoint_tuple and "messages" in checkpoint_tuple.checkpoint["channel_values"]:
                return checkpoint_tuple.checkpoint["channel_values"]["messages"]
        except Exception as e:
            logger.warning(f"Could not retrieve message history for session {session_id}: {e}")
        return []

    async def process_query(self, query: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Process a user query using the LangGraph workflow, maintaining full conversation history."""
        # Generate IDs if not provided
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

        logger.info(f"Processing query for user {user_id}, session {session_id}")
        logger.info(f"Query: {query}")

        # Retrieve previous message history for this session
        message_history = self._get_message_history_from_memory(session_id)
        message_history = list(message_history)  # Defensive copy
        message_history.append(HumanMessage(content=query))

        # Create initial state with full message history
        initial_state = {
            "messages": message_history,
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "elasticsearch_queries": [],
            "search_results": [],
            "query_analysis": {},
            "step_count": 0,
            "max_steps": self.config.max_steps,
            "reasoning_steps": [],
            "final_answer": "",
            "metadata": {},
            "search_needed": True,
            "conversational_response": "",
            "contextualized_query": query
        }

        # Configuration for the run
        config = RunnableConfig(
            configurable={"thread_id": session_id},
            tags=["elasticsearch", "agent", user_id]
        )

        try:
            # Execute the workflow
            logger.info("Starting workflow execution...")
            final_state = await self.graph.ainvoke(initial_state, config=config)

            # Extract results
            result = {
                "answer": final_state["final_answer"],
                "metadata": final_state["metadata"],
                "queries_executed": final_state.get("elasticsearch_queries", []),
                "reasoning_steps": final_state["reasoning_steps"],
                "session_id": session_id,
                "user_id": user_id,
                # Return updated message history for external use if needed
                "messages": final_state["messages"]
            }

            logger.info("Query processing completed successfully")
            logger.info(result)
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "user_id": user_id
            }
    
    async def interactive_chat(self, user_id: str = None):
        """Interactive chat interface"""
        user_id = user_id or f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        print(f"\n🤖 Enhanced LangGraph Elasticsearch Agent Started!")
        print(f"👤 User ID: {user_id}")
        print(f"🔐 Session ID: {session_id}")
        print("🔍 Features: Multi-step reasoning, intelligent query building, persistent memory")
        print("📊 Specialized for Elasticsearch queries and analytics")
        print("Type your questions or 'quit' to exit.\n")
        
        while True:
            try:
                query = input("🔍 You: ").strip()
                
                if query.lower() in ["quit", "exit", "bye"]:
                    print("👋 Goodbye! Your conversation has been saved.")
                    break
                
                if not query:
                    continue
                
                print("🔄 Processing query...")
                print("📋 Analyzing intent and planning steps...")
                
                result = await self.process_query(query, user_id, session_id)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                print(f"\n🤖 Agent: {result['answer']}")
                
                # Show execution details
                metadata = result.get("metadata", {})
                print(f"\n📊 Execution Details:")
                print(f"   Steps: {metadata.get('total_steps', 0)}")
                print(f"   Queries: {metadata.get('queries_executed', 0)}")
                print(f"   Reasoning Steps: {metadata.get('reasoning_steps_count', 0)}")
                
                # Show queries that were built
                queries = result.get("queries_executed", [])
                if queries:
                    print(f"\n🔍 Elasticsearch Queries Built:")
                    for i, q in enumerate(queries, 1):
                        print(f"   {i}. {q['query']['query_type']}: {q['query']['purpose']}")
                        print(f"   Query Body: {json.dumps(q['query']['query_body'], indent=2)}")
                
                # Print raw search results for debugging
                results = result.get('search_results', [])
                if results:
                    print(f"\n🔎 Raw Search Results:")
                    for r in results:
                        print(json.dumps(r, indent=2))
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Your conversation has been saved.")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logger.error(f"Interactive chat error: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, '_session_context'):
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, '_stdio_context'):
                await self._stdio_context.__aexit__(None, None, None)
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LangGraph Elasticsearch Agent")
    parser.add_argument("--mcp-port", type=int, default=8090, help="MCP server port")
    parser.add_argument("--mcp-host", type=str, default="localhost", help="MCP server host")
    parser.add_argument("--user-id", type=str, help="User ID for session")
    parser.add_argument("--interactive", action="store_true", default=True, help="Run interactive mode")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum reasoning steps")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    
    args = parser.parse_args()
    
    # Create agent config
    config = AgentConfig(
        max_steps=args.max_steps,
        temperature=args.temperature,
        enable_console_logging=True
    )
    
    agent = LangGraphElasticsearchAgent(config)
    
    try:
        # Connect to MCP server
        server_command = "uv"
        server_args = ["run", "elasticsearch-mcp-server"]
        import os
        es_env = {
            "ELASTICSEARCH_HOSTS": os.getenv("ELASTICSEARCH_HOSTS", "https://localhost:9200"),
            "ELASTICSEARCH_USERNAME": os.getenv("ELASTICSEARCH_USERNAME", "elastic"),
            "ELASTICSEARCH_PASSWORD": os.getenv("ELASTICSEARCH_PASSWORD", "test123"),
            **os.environ,
        }
        await agent.connect_to_mcp_server(server_command, server_args, env=es_env)
        
        if args.interactive:
            await agent.interactive_chat(args.user_id)
        else:
            # Example query for testing
            result = await agent.process_query(
                "Show me the top 10 error messages from the last 24 hours", 
                args.user_id or "test_user"
            )
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Main error: {e}")
        print(f"❌ Error: {str(e)}")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
