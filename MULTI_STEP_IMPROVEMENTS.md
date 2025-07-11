# Multi-Step Query Processing Improvements

## Problem
The agent was not properly identifying complex queries that require multi-step processing. For example, queries like:
- "summarize from all projects what we have done at DBS Bank Ltd account"
- "compare the success criteria of Department of Defence project with Westpac project"

These queries were being treated as single-step searches, leading to poor results and failed Elasticsearch queries.

## Solution
Enhanced the query analysis and execution pipeline to properly identify and handle complex multi-step queries.

### 1. Improved Query Analysis (`_analyze_query_node`)

**Enhanced Guidelines for Multi-Step Detection:**
- Added specific patterns that trigger multi-step processing:
  - "summarize from all projects" or "summarize across projects"
  - "what we have done" or "work done" or "accomplishments"
  - "compare" or "comparison" between different entities
  - "find all" + "then" + another action
  - "list" + "and" + "analyze" or "summarize"
  - Queries with multiple distinct information needs
  - Queries requiring data gathering followed by analysis/synthesis

**Complexity Guidelines:**
- **simple**: Single search term, direct lookup
- **moderate**: Multiple search terms, filtering, basic aggregation
- **complex**: Multi-step reasoning, data gathering + analysis, comparisons, summarization

### 2. Enhanced Step Planning (`_plan_steps_node`)

**Multi-Step Planning Guidelines:**
- **For "summarize from all projects" queries:**
  1. First step: Find all projects matching the criteria (e.g., "DBS Bank Ltd")
  2. Second step: Extract work done/accomplishments from those projects
  3. Third step: Summarize and synthesize the findings

- **For comparison queries:**
  1. First step: Gather data for first entity
  2. Second step: Gather data for second entity
  3. Third step: Compare and analyze differences

- **For "what we have done" queries:**
  1. First step: Find relevant projects/entities
  2. Second step: Extract accomplishments/work done
  3. Third step: Summarize the work

### 3. Context-Aware Step Execution (`_execute_step_node`)

**New Feature: Contextual Query Building**
- Added `_build_contextual_step_query()` method
- Uses results from previous steps to build more specific queries for subsequent steps
- For example: If step 1 finds specific project names, step 2 searches for work done in those specific projects

### 4. Improved Answer Synthesis (`_synthesize_answer_node`)

**Enhanced Synthesis Guidelines:**
- Follows the logical flow of execution steps
- Uses results from each step to build a coherent narrative
- For project summaries: Lists projects found, then summarizes work done across those projects
- For comparisons: Presents data for each entity, then highlights key differences
- For work summaries: Identifies entities/projects, then summarizes accomplishments

## Example Multi-Step Flow

**Query:** "summarize from all projects what we have done at DBS Bank Ltd account"

**Step 1:** Find all projects related to "DBS Bank Ltd"
- Query: "DBS Bank Ltd projects"
- Result: List of project names and details

**Step 2:** Extract work done/accomplishments from those projects
- Query: "work done accomplishments DBS Bank Ltd [specific project names from step 1]"
- Result: Specific work items and achievements

**Step 3:** Summarize and synthesize findings
- Combines results from steps 1 and 2
- Provides comprehensive summary of work done for DBS Bank Ltd

## Testing

Use the `test_multi_step.py` script to verify that complex queries are properly identified and planned:

```bash
python test_multi_step.py
```

This will test various query patterns and show whether multi-step processing is triggered correctly.

## Benefits

1. **Better Query Understanding**: Agent now properly identifies complex queries that need multi-step processing
2. **Improved Results**: Multi-step execution leads to more comprehensive and accurate answers
3. **Context Awareness**: Subsequent steps use results from previous steps for better targeting
4. **Structured Responses**: Final answers follow logical flow and provide better synthesis
5. **Reduced Query Failures**: Complex queries are broken down into manageable, targeted searches

## Configuration

The multi-step processing can be configured via:
- `max_steps`: Maximum number of execution steps (default: 10)
- `temperature`: LLM temperature for analysis and planning (default: 0.1)
- Query analysis patterns can be modified in the `_analyze_query_node` method 
