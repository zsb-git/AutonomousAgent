"""Stage 1 — the agent loop, by hand.

No planning, no memory, no retries yet - just the core mechanic every agent
is built on:

    LLM decides an action -> a tool executes it -> the LLM observes the
    result -> repeat until the LLM decides it's done.

This is the "ReAct loop" you'll see named everywhere. LangGraph's job here is
just to make that loop an explicit graph instead of a hidden while-loop.
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


@tool
def web_search(query: str) -> str:
    """Search the web and return a few short result snippets for the query."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

    # DuckDuckGo mixes sponsored results in with organic ones, marked by a
    # title that starts with "Ad" - skip those, we only want real content.
    organic = [r for r in results if not r.get("title", "").startswith("Ad")]

    if not organic:
        return "No results found."

    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in organic)


# --- State -----------------------------------------------------------------
# For Stage 1, state is just the running message history. `add_messages` is a
# LangGraph reducer: instead of each node's return value *replacing*
# state["messages"], it *appends* to it. That's the only thing making this
# a conversation instead of a series of unrelated calls.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- Nodes -------------------------------------------------------------------
tools = [web_search]
llm = ChatAnthropic(model="claude-haiku-5-5").bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """Ask the LLM what to do next, given everything that's happened so far."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """The routing decision: did the LLM just ask to call a tool, or is it done?"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# --- Graph -------------------------------------------------------------------
#
#        START
#          |
#          v
#      [ agent ] <-------+
#          |              |
#   tool_calls? ----------+
#      yes |  no
#          v   \
#      [ tools ] \-> END
#
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()


def extract_text(content) -> str:
    """Claude's responses can be a plain string, or a list of content blocks
    (e.g. an internal 'thinking' block alongside the actual 'text' block).
    We only want the text parts for display."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def main() -> None:
    goal = (
        "What is a recent, specific SAP technology announcement or product "
        "update? Name it and give a one-paragraph summary."
    )
    print(f"Goal: {goal}\n")

    result = graph.invoke({"messages": [("user", goal)]})

    print("--- Full trace (every message in order) ---\n")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        tool_calls = getattr(msg, "tool_calls", None)
        text = extract_text(msg.content)
        print(f"[{role}]")
        if tool_calls:
            for call in tool_calls:
                print(f"  -> calling tool: {call['name']}({call['args']})")
        if text:
            print(f"  {text[:400]}")
        print()

    print("--- Final answer ---\n")
    print(extract_text(result["messages"][-1].content))


if __name__ == "__main__":
    main()
