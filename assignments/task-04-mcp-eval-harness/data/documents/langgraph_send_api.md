# LangGraph Send() API and Parallel Execution

LangGraph's `Send()` API enables dynamic, data-driven fan-out within a compiled graph. Instead of statically wiring edges between nodes, `Send()` lets a conditional edge function return a list of `Send` objects at runtime, each targeting the same or different nodes with different payloads.

## How Send() Works

A `Send` object wraps two things: the name of the destination node and the state dictionary that node should receive. When a conditional edge returns multiple `Send` objects, LangGraph schedules all of them in parallel. This is fundamentally different from a static edge, which always routes to exactly one node.

```python
from langgraph.types import Send

def dispatch_workers(state: MyState) -> list[Send]:
    return [Send("worker_node", {**state, "task": t}) for t in state["tasks"]]
```

## Fan-In with Reducers

Parallel branches all write back to the shared state. To prevent branches from overwriting each other's results, LangGraph requires a reducer annotation on fields that multiple branches write to. The most common reducer is `operator.add`, which concatenates lists:

```python
from typing import Annotated
import operator

class MyState(TypedDict):
    results: Annotated[list, operator.add]
```

When each parallel worker appends to `results`, LangGraph's runtime automatically merges all contributions using the reducer before passing the combined state to the next node.

## Use Cases

The Send() API is ideal for map-reduce patterns: splitting a large problem into independent sub-problems (map), solving each in parallel, and then aggregating the results (reduce). Common applications include parallel document retrieval, multi-agent research pipelines, and batch data processing within a single graph run.

## Convergence

All `Send()` branches converge at a single "fan-in" node. LangGraph waits for every parallel branch to complete before invoking the convergence node. This ensures the fan-in node always receives the fully merged state from all branches, making the parallel execution transparent to downstream logic.
