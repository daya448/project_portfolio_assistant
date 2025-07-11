#!/usr/bin/env python3
"""
Test script to verify multi-step query analysis improvements
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from enhanced_agent_manager import LangGraphElasticsearchAgent, AgentConfig

load_dotenv()

async def test_query_analysis():
    """Test the query analysis for complex queries"""
    
    # Initialize agent
    config = AgentConfig(
        max_steps=10,
        temperature=0.1,
        enable_console_logging=True
    )
    agent = LangGraphElasticsearchAgent(config)
    
    # Test queries that should trigger multi-step processing
    test_queries = [
        "summarize from all projects what we have done at DBS Bank Ltd account",
        "what work have we completed for Westpac project",
        "compare the success criteria of Department of Defence project with Westpac project",
        "list all projects and analyze their current status",
        "find all projects related to banking and summarize their accomplishments"
    ]
    
    print("Testing Query Analysis for Multi-Step Detection\n")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        
        try:
            # Create a mock state for testing
            state = {
                "query": query,
                "messages": [],
                "user_id": "test_user",
                "session_id": "test_session",
                "elasticsearch_queries": [],
                "search_results": [],
                "query_analysis": {},
                "step_count": 0,
                "max_steps": 10,
                "reasoning_steps": [],
                "final_answer": "",
                "metadata": {},
                "search_needed": True,
                "conversational_response": "",
                "contextualized_query": query
            }
            
            # Test the analysis node
            result_state = await agent._analyze_query_node(state)
            
            analysis = result_state["query_analysis"]
            print(f"Complexity: {analysis.get('complexity', 'unknown')}")
            print(f"Requires Multi-Step: {analysis.get('requires_multi_step', False)}")
            print(f"Intent: {analysis.get('intent', 'unknown')}")
            print(f"Key Concepts: {analysis.get('key_concepts', [])}")
            print(f"Elasticsearch Strategy: {analysis.get('elasticsearch_strategy', 'unknown')}")
            
            # Test the planning node
            if analysis.get('requires_multi_step', False):
                print("\n✅ Multi-step processing triggered!")
                plan_state = await agent._plan_steps_node(result_state)
                steps = plan_state["reasoning_steps"]
                print(f"Planned {len(steps)} steps:")
                for j, step in enumerate(steps, 1):
                    print(f"  Step {j}: {step['description']} ({step['action']})")
            else:
                print("\n❌ Multi-step processing NOT triggered")
                
        except Exception as e:
            print(f"Error testing query: {e}")
        
        print()

if __name__ == "__main__":
    asyncio.run(test_query_analysis()) 
