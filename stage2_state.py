"""Stage 2 - making state explicit, and watching it evolve.

Stage 1's state was really just "the chat history" - everything the agent
knew was implicitly buried inside the messages list. That works, but it
means the *graph itself* can't easily reason about anything that isn't a
message (how many steps has this taken? what was the original goal, as
opposed to whatever's currently being discussed?).

This stage adds two explicit fields to state - `goal` and `iterations` - and
uses `iterations` for something a message list can't easily give you: a hard
safety cap on how many times the loop can run, enforced by the graph itself,
independent of whether the LLM "wants" to keep going.

It also switches from `.invoke()` (run everything, get the final result) to
`.stream()` (get the state back after every single step), so you can watch
each field change in real time instead of only seeing the end result.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

MAX_ITERATIONS = 4  # a hard cap, enforced by the graph - not a suggestion to the LLM


@tool
def web_search(query: str) -> str:
    """Search the web and return a few short result snippets for the query."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
    organic = [r for r in results if not r.get("title", "").startswith("Ad")]
    if not organic:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in organic)


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


# --- State -------------------------------------------------------------------
# `messages` still uses the add_messages reducer (append). `goal` and
# `iterations` have NO reducer, which means: whatever a node returns for that
# key *replaces* the previous value. That's the right default for a counter
# you're explicitly recomputing, or a goal that's set once and never changes.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    goal: str
    iterations: int


tools = [web_search]
llm = ChatAnthropic(model="claude-haiku-5-5").bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    # Reading AND writing a specific state field, not just appending to
    # messages - this is the thing Stage 1 couldn't really do.
    return {"messages": [response], "iterations": state["iterations"] + 1}


def should_continue(state: AgentState) -> str:
    # The iteration cap is checked FIRST and wins regardless of what the LLM
    # wants - this is the graph enforcing a boundary the LLM doesn't control.
    if state["iterations"] >= MAX_ITERATIONS:
        return END
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()


def main() -> None:
    goal = (
        "What is a recent, specific SAP technology announcement or product "
        "update? Name it and give a one-paragraph summary."
    )
    initial_state: AgentState = {
        "messages": [("user", goal)],
        "goal": goal,
        "iterations": 0,
    }

    print(f"Goal: {goal}")
    print(f"(hard cap: {MAX_ITERATIONS} agent iterations)\n")
    print("--- Watching state after every step ---\n")

    final_state = None
    for state_snapshot in graph.stream(initial_state, stream_mode="values"):
        final_state = state_snapshot
        last_msg = state_snapshot["messages"][-1]
        role = last_msg.__class__.__name__
        tool_calls = getattr(last_msg, "tool_calls", None)

        print(f"[iterations={state_snapshot['iterations']}] latest message: {role}")
        if tool_calls:
            for call in tool_calls:
                print(f"  -> calling tool: {call['name']}({call['args']})")
        else:
            text = extract_text(last_msg.content)
            if text:
                print(f"  {text[:200]}")
        print()

    print("--- Final state ---")
    print(f"goal: {final_state['goal']}")
    print(f"iterations used: {final_state['iterations']} / {MAX_ITERATIONS}")
    print()
    print("--- Final answer ---\n")
    print(extract_text(final_state["messages"][-1].content))


if __name__ == "__main__":
    main()
