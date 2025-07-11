#!/usr/bin/env python3
"""
Test script to verify general multi-step query analysis improvements
Tests queries that don't rely on specific keywords but are still complex
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from enhanced_agent_manager import LangGraphElasticsearchAgent, AgentConfig

load_dotenv()

async def test_general_query_analysis():
    """Test the query analysis for general complex queries"""
    
    # Initialize agent
    config = AgentConfig(
        max_steps=10,
        temperature=0.1,
        enable_console_logging=True
    )
    agent = LangGraphElasticsearchAgent(config)
    
    # Test queries that are complex but don't use specific keywords
    test_queries = [
        # Timeline and temporal queries
        "Show me the timeline of all banking initiatives and their outcomes",
        "What are the key deliverables across our enterprise projects?",
        "Analyze the risk factors for our major clients",
        
        # Hierarchical and categorical queries
        "Break down our project portfolio by industry sector and show performance metrics",
        "What are the success rates of projects in different technology domains?",
        "Show me the distribution of project budgets across different client types",
        
        # Sequential reasoning queries
        "Find all projects that started in Q1 and analyze their current status",
        "Identify projects with high risk scores and show their mitigation strategies",
        "List all completed projects and summarize their key learnings",
        
        # Multi-dimensional analysis
        "What are the common challenges across projects in the financial services sector?",
        "Analyze the correlation between project duration and client satisfaction scores",
        "Show me the impact of team size on project delivery timelines",
        
        # Complex aggregation queries
        "What percentage of our projects are delivered on time, and what factors contribute to delays?",
        "Break down our project portfolio by complexity level and show success metrics for each",
        "Analyze the relationship between project scope changes and delivery performance"
    ]
    
    print("Testing General Query Analysis for Multi-Step Detection\n")
    print("=" * 80)
    
    multi_step_count = 0
    total_queries = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 60)
        
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
            requires_multi_step = analysis.get('requires_multi_step', False)
            complexity = analysis.get('complexity', 'unknown')
            
            print(f"Complexity: {complexity}")
            print(f"Requires Multi-Step: {requires_multi_step}")
            print(f"Intent: {analysis.get('intent', 'unknown')}")
            print(f"Key Concepts: {analysis.get('key_concepts', [])}")
            print(f"Strategy: {analysis.get('elasticsearch_strategy', 'unknown')}")
            
            if requires_multi_step:
                multi_step_count += 1
                print("\n✅ Multi-step processing triggered!")
                
                # Test the planning node
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
    
    # Summary
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"Total queries tested: {total_queries}")
    print(f"Multi-step triggered: {multi_step_count}")
    print(f"Single-step queries: {total_queries - multi_step_count}")
    print(f"Multi-step detection rate: {(multi_step_count/total_queries)*100:.1f}%")
    
    if multi_step_count / total_queries >= 0.7:
        print("✅ Good multi-step detection rate!")
    elif multi_step_count / total_queries >= 0.5:
        print("⚠️ Moderate multi-step detection rate - may need improvement")
    else:
        print("❌ Low multi-step detection rate - needs significant improvement")

if __name__ == "__main__":
    asyncio.run(test_general_query_analysis()) 
