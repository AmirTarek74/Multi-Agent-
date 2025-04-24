from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from typing import Literal
from langgraph.types import Command
from .tools import tavily_tool, scrape_webpages
from utils.supervisor import State, make_supervisor_node
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()  
google_api_key = os.getenv("GOOGLE_API_KEY")
llm_name = "gemini-1.5-flash-8b"
llm = ChatGoogleGenerativeAI(model=llm_name, api_key=google_api_key, verbose=False)


search_agent = create_react_agent(llm, tools=[tavily_tool])

def search_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Calls the search agent to search for documents based on the user's query, and routes to the supervisor when done.

    Args:
        state (State): The state of the graph.

    Returns:
        Command[Literal["supervisor"]]: A command to route to the supervisor when done.
    """

    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])



def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Calls the web scraper agent to scrape the web based on the user's query, and routes to the supervisor when done.

    Args:
        state (State): The state of the graph.

    Returns:
        Command[Literal["supervisor"]]: A command to route to the supervisor when done.
    """
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

# Define the graph.
research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()


def call_research_team(state: State) -> Command[Literal["supervisor"]]:
        """
        Calls the research team (search and web scraper) to generate a response to the user's query, and routes to the supervisor when done.

        Args:
            state (State): The state of the graph.

        Returns:
            Command[Literal["supervisor"]]: A command to route to the supervisor when done.
        """

        response = research_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="research_team"
                    )
                ]
            },
            goto="supervisor",
        )