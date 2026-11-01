"""Stage 3 - planning (task decomposition).

Stages 1-2 were one open-ended loop: the LLM searched and reasoned until it
felt done, with no explicit structure to *what* it was doing along the way.

This stage adds a `planner` node that turns the goal into an explicit,
ordered list of subtasks (structured output - a real Pydantic object, not
text you have to parse). The graph then runs the Stage-2 agent loop once per
subtask, collects each subtask's result, and finally a `compose` node
stitches everything into one answer.

The whole run shares ONE growing message list - each new subtask is
introduced as a new message, so the agent keeps whatever context it
accumulated from earlier subtasks. Simpler than resetting state per step,
and arguably more useful: it won't repeat research it already did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from emailer import GMAIL_ADDRESS, send_email

load_dotenv()


@tool
def web_search(query: str) -> str:
    """Search the web and return a few short result snippets for the query."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
    except Exception as exc:
        return f"Search failed for query '{query}': {exc}"
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


# --- Structured output for the planner ---------------------------------------
# A Pydantic model, not a string you'd have to parse yourself. The LLM is
# forced (via tool-calling under the hood) to return exactly this shape.
class Plan(BaseModel):
    steps: list[str] = Field(
        description=(
            "3 to 5 concrete, ordered subtasks. Each should be independently "
            "researchable (e.g. answerable via a web search) and together "
            "they should fully accomplish the goal."
        )
    )


# --- State ---------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    goal: str
    plan: list[str]
    current_step: int
    results: list[str]
    final_answer: str


tools = [web_search]
llm = ChatAnthropic(model="claude-haiku-5-5").bind_tools(tools)
planner_llm = ChatAnthropic(model="claude-haiku-5-5").with_structured_output(Plan)
compose_llm = ChatAnthropic(model="claude-haiku-5-5")


def planner_node(state: AgentState) -> dict:
    plan: Plan = planner_llm.invoke(
        f"Break the following goal into subtasks.\n\nGoal: {state['goal']}"
    )
    intro = f"Subtask 1 of {len(plan.steps)}: {plan.steps[0]}"
    return {
        "plan": plan.steps,
        "current_step": 0,
        "results": [],
        "messages": [("user", intro)],
    }


def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "advance"  # this step is done - go record the result and move on


def advance_node(state: AgentState) -> dict:
    """Current subtask just finished (the agent stopped calling tools).
    Record its result, and either introduce the next subtask or signal
    we're done."""
    completed_result = extract_text(state["messages"][-1].content)
    results = state["results"] + [completed_result]
    next_step = state["current_step"] + 1

    update: dict = {"results": results, "current_step": next_step}

    if next_step < len(state["plan"]):
        task = state["plan"][next_step]
        intro = f"Subtask {next_step + 1} of {len(state['plan'])}: {task}"
        update["messages"] = [("user", intro)]

    return update


def route_after_advance(state: AgentState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "agent"
    return "compose"


def compose_node(state: AgentState) -> dict:
    steps_and_results = "\n\n".join(
        f"Subtask {i + 1}: {task}\nResult: {result}"
        for i, (task, result) in enumerate(zip(state["plan"], state["results"]))
    )
    prompt = (
        f"Original goal: {state['goal']}\n\n"
        f"Findings from each subtask:\n\n{steps_and_results}\n\n"
        "Write one cohesive, concrete answer to the original goal using "
        "these findings. Don't just list the subtasks - synthesize them."
    )
    response = compose_llm.invoke(prompt)
    return {"final_answer": extract_text(response.content)}


# --- Graph ---------------------------------------------------------------
#
#   START -> planner -> agent <--> tools      (inner loop, per subtask)
#                          |
#                     (no tool calls)
#                          v
#                      advance --> agent        (more subtasks left)
#                          |
#                   (plan exhausted)
#                          v
#                      compose -> END
#
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("advance", advance_node)
builder.add_node("compose", compose_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "advance": "advance"})
builder.add_edge("tools", "agent")
builder.add_conditional_edges("advance", route_after_advance, {"agent": "agent", "compose": "compose"})
builder.add_edge("compose", END)

graph = builder.compile()

OUTPUT_FILE = Path(__file__).parent / "stage3_output.txt"


def main() -> None:
    goal = (
        "Research the best technology news in the SAP space, give pointers "
        "of what it is, and an example describing it."
    )
    initial_state: AgentState = {
        "messages": [],
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "results": [],
        "final_answer": "",
    }

    # Full detail goes to a file (OUTPUT_FILE); the terminal only gets short
    # progress lines so you can tell it's alive without scrolling through
    # the whole trace every run.
    log_lines: list[str] = [f"Goal: {goal}", ""]

    def log(line: str = "") -> None:
        log_lines.append(line)

    print(f"Goal: {goal}")
    print(f"(full trace will be written to {OUTPUT_FILE.name})\n")

    final_state = None
    seen_plan = False
    last_message_count = 0
    for state_snapshot in graph.stream(initial_state, stream_mode="values"):
        final_state = state_snapshot

        if state_snapshot["plan"] and not seen_plan:
            seen_plan = True
            log("--- Plan ---")
            for i, task in enumerate(state_snapshot["plan"], start=1):
                log(f"  {i}. {task}")
            log()
            print(f"Plan: {len(state_snapshot['plan'])} subtasks")

        messages = state_snapshot["messages"]
        if len(messages) <= last_message_count:
            # advance/compose ran without adding a new message (e.g. right
            # after the last subtask finishes) - nothing new to log.
            continue
        last_message_count = len(messages)

        last_msg = messages[-1]
        role = last_msg.__class__.__name__
        tool_calls = getattr(last_msg, "tool_calls", None)
        step_num = min(state_snapshot["current_step"] + 1, len(state_snapshot["plan"]))

        line = f"[subtask {step_num}] {role}"
        if tool_calls:
            for call in tool_calls:
                line += f" -> calling tool: {call['name']}({call['args']})"
            print(f"  subtask {step_num}: searching ({len(tool_calls)} call(s))...")
        else:
            text = extract_text(last_msg.content)
            if text:
                line += f": {text}"
            print(f"  subtask {step_num}: done")
        log(line)

    log("\n--- Results per subtask ---")
    for i, (task, result) in enumerate(zip(final_state["plan"], final_state["results"]), start=1):
        log(f"\n{i}. {task}")
        log(f"   {result}")

    log("\n--- Final answer ---\n")
    log(final_state["final_answer"])

    OUTPUT_FILE.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nDone. Full output written to {OUTPUT_FILE}")

    if GMAIL_ADDRESS:
        send_email(
            subject=f"Autonomous Agent results: {goal[:60]}",
            body=f"Goal: {goal}\n\n{final_state['final_answer']}\n\nFull trace attached.",
            to_address=GMAIL_ADDRESS,
            attachment_path=OUTPUT_FILE,
        )


if __name__ == "__main__":
    main()
