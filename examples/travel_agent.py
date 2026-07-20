"""A LangGraph travel-assistant agent with several tools, instrumented via ``EqtyCallbackHandler``.

The graph is the same agent/tools loop as langgraph_example.py, but with a larger tool belt so the
lineage graph shows multiple distinct Tool assets and tool computations:

    START -> agent -> (tool calls?) -> tools -> agent -> ... -> summarize -> END

Tools available to the model:

- get_weather        -> mock current conditions for a city
- list_attractions   -> mock sights for a city
- get_exchange_rate  -> mock currency rates
- calculate          -> basic arithmetic (e.g. converting a budget with a rate)

Pass ``--debug`` to enable the handler's debug flag, which attaches extra metadata to the
registered EQTY assets.

Requires OPENAI_API_KEY.

    uv run python examples/multi_tool_example.py "your question" [--debug]
"""

import argparse
import os
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from eqty_sdk import Context, Signer, init, set_active_signer

from eqty_lineage.langchain import EqtyCallbackHandler, eqty_tool

WEATHER = {
    "amsterdam": "14°C, light rain, wind 20 km/h",
    "lisbon": "24°C, sunny, wind 10 km/h",
    "tokyo": "19°C, overcast, wind 5 km/h",
}

ATTRACTIONS = {
    "amsterdam": ["Rijksmuseum", "Van Gogh Museum", "Canal cruise"],
    "lisbon": ["Belém Tower", "Alfama district", "Tram 28"],
    "tokyo": ["Senso-ji", "Meiji Shrine", "teamLab Planets"],
}

EXCHANGE_RATES = {
    ("usd", "eur"): 0.92,
    ("usd", "jpy"): 155.0,
    ("eur", "usd"): 1.09,
    ("eur", "jpy"): 168.0,
}

SYSTEM_PROMPT = (
    "You are a travel assistant. Use get_weather and list_attractions to look up "
    "destination info, get_exchange_rate for currency rates, and calculate for any "
    "arithmetic such as converting a budget. Answer concisely."
)


@tool
@eqty_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    conditions = WEATHER.get(city.strip().lower())
    if conditions is None:
        return f"No weather data for '{city}'. Known cities: {', '.join(WEATHER)}."
    return f"Weather in {city}: {conditions}"


@tool
@eqty_tool
def list_attractions(city: str) -> str:
    """List the top attractions for a city."""
    sights = ATTRACTIONS.get(city.strip().lower())
    if sights is None:
        return f"No attraction data for '{city}'. Known cities: {', '.join(ATTRACTIONS)}."
    return f"Top attractions in {city}: {', '.join(sights)}."


@tool
@eqty_tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the exchange rate between two currencies, e.g. from_currency='USD', to_currency='EUR'."""
    pair = (from_currency.strip().lower(), to_currency.strip().lower())
    rate = EXCHANGE_RATES.get(pair)
    if rate is None:
        known = ", ".join(f"{a.upper()}->{b.upper()}" for a, b in EXCHANGE_RATES)
        return f"No rate for {from_currency.upper()}->{to_currency.upper()}. Known pairs: {known}."
    return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}"


@tool
@eqty_tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '1500 * 0.92'."""
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return "Error: only digits and + - * / ( ) are supported."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - input is character-whitelisted
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


TOOLS = [get_weather, list_attractions, get_exchange_rate, calculate]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    answer: str


def build_graph(model=None):
    if model is None:
        model = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=2048,
        ).bind_tools(TOOLS)

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "summarize"

    def summarize(state: AgentState) -> dict:
        """Deterministic (non-LLM) node: extract the final answer from the transcript."""
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        return {"answer": text}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "summarize"])
    graph.add_edge("tools", "agent")
    graph.add_edge("summarize", END)
    return graph.compile()


def init_sdk():
    ctx = Context.new("Travel Assistant")
    cfg = init(default_context=ctx).set_store_all_blobs(True)
    signer = Signer.new(name="travel_assistant", _load_if_exists=True)
    set_active_signer(signer)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EQTY-instrumented travel assistant.")
    parser.add_argument(
        "question",
        nargs="?",
        default=(
            "I'm choosing between Amsterdam and Lisbon this weekend. Compare the weather "
            "and attractions, and convert my 1500 USD budget to EUR."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable EqtyCallbackHandler debug mode (extra metadata on registered assets).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_graph()
    result = app.invoke(
        {"messages": [HumanMessage(args.question)], "question": args.question, "answer": ""},
        config={"callbacks": [EqtyCallbackHandler()], "recursion_limit": 25},
    )
    print(result["answer"])


if __name__ == "__main__":
    cfg = init_sdk()
    main()
    ctx = cfg.get_default_context()
    ctx.export(Path("./manifests/travel-assistant.json"))
