---
sidebar_position: 12
---

# Search Agent

The Search Agent enables flows to retrieve information from external sources. It supports two primary search modes: live web search and semantic search over your private documents.

### Providers

The agent routes its execution based on the parameters provided in `model_params`:

1.  **Google**: Live web search using the Google Custom Search JSON API.
2.  **Intellicloud**: Semantic search over data indexed in the Intellicloud platform.

### Parameters

- **k**: Number of search results to return (default: 5 for Google, 3 for Intellicloud).
- **as_text**: (Google only) If `True`, returns a formatted string. If `False`, returns a structured list of results.

---

### Example: Google Web Search

To use Google search, you need a **Google API Key** and a **Custom Search Engine ID (CX)**.

```python
import asyncio
from intelli.flow.agents.agent import Agent
from intelli.flow.types import AgentTypes
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.sequence_flow import SequenceFlow

# 1. Define the Google Search Agent
search_agent = Agent(
    agent_type=AgentTypes.SEARCH.value,
    provider="google",
    mission="find latest news about AI",
    model_params={
        "google_api_key": "YOUR_GOOGLE_API_KEY",
        "google_cse_id": "YOUR_CSE_ID",
        "k": 3,
        "as_text": True
    }
)

# 2. Create and run the flow
task = Task(TextTaskInput("What is the latest update on GPT-5.2?"), search_agent)
flow = SequenceFlow([task])
result = flow.start()

print(result["task1"])
```

---

### Example: Intellicloud Semantic Search

This mode searches through your own documents previously indexed via Intellicloud.

```python
# Define the Intellicloud Search Agent
search_agent = Agent(
    agent_type=AgentTypes.SEARCH.value,
    provider="intellicloud",
    mission="search in my documentation",
    model_params={
        "one_key": "YOUR_INTELLICLOUD_ONE_KEY",
        "k": 3
    }
)

# Execution follows the same pattern as above
```

### Notes

- **Input**: The Search Agent always expects a text-based query as input.
- **Output**: By default, it returns a text summary of the results, making it easy to feed into a subsequent LLM task for analysis.

