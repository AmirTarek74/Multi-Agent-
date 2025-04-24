from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from typing import Literal
from langgraph.types import Command
from .tools import *
from utils.supervisor import State, make_supervisor_node
from langgraph.graph import StateGraph, MessagesState, START, END
from .tools import *
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()  
google_api_key = os.getenv("GOOGLE_API_KEY")
llm_name = "gemini-1.5-flash-8b"
llm = ChatGoogleGenerativeAI(model=llm_name, api_key=google_api_key, verbose=False)


doc_writer_agent = create_react_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    prompt=(
        "You can read, write and edit documents based on note-taker's outlines. "
        "Don't ask follow-up questions."
    ),
)


def doc_writing_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Calls the doc writer agent to generate a document based on the user's query and the note-taker's outline, and routes to the supervisor when done.

    Args:
        state (State): The state of the graph.

    Returns:
        Command[Literal["supervisor"]]: A command to route to the supervisor when done.
    """
    
    result = doc_writer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="doc_writer")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


note_taking_agent = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    prompt=(
        "You can read documents and create outlines for the document writer. "
        "Don't ask follow-up questions."
    ),
)


def note_taking_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Calls the note-taking agent to read documents and create outlines for the document writer based on the user's query, and routes to the supervisor when done.

    Args:
        state (State): The state of the graph.

    Returns:
        Command[Literal["supervisor"]]: A command to route to the supervisor when done.
    """

    result = note_taking_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="note_taker")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


chart_generating_agent = create_react_agent(
    llm, tools=[read_document, python_repl_tool]
)


def chart_generating_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Calls the chart generating agent to generate a chart based on the user's query and the document writer's document, and routes to the supervisor when done.

    Args:
        state (State): The state of the graph.

    Returns:
        Command[Literal["supervisor"]]: A command to route to the supervisor when done.
    """
    result = chart_generating_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="chart_generator"
                )
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


doc_writing_supervisor_node = make_supervisor_node(
    llm, ["doc_writer", "note_taker", "chart_generator"]
)

# Define the graph.
paper_writing_builder = StateGraph(State)
paper_writing_builder.add_node("supervisor", doc_writing_supervisor_node)
paper_writing_builder.add_node("doc_writer", doc_writing_node)
paper_writing_builder.add_node("note_taker", note_taking_node)
paper_writing_builder.add_node("chart_generator", chart_generating_node)

paper_writing_builder.add_edge(START, "supervisor")
paper_writing_graph = paper_writing_builder.compile()

def call_paper_writing_team(state: State) -> Command[Literal["supervisor"]]:
        
        """
        Calls the paper writing team (document writer, note-taker, and chart generator) to generate a document based on the user's query, and routes to the supervisor when done.

        Args:
            state (State): The state of the graph.

        Returns:
            Command[Literal["supervisor"]]: A command to route to the supervisor when done.
        """

        response = paper_writing_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="writing_team"
                    )
                ]
            },
            goto="supervisor",
        )