import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.supervisor import State
from search.agents import *
from write.agents import *
import argparse


def main(question):
    """
    Main function that takes a question as input and runs the graph to generate the output.

    Args:
        question (str): The question to ask the graph.

    Returns:
        None
    """
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
    llm_name = "gemini-1.5-flash-8b"
    llm = ChatGoogleGenerativeAI(model=llm_name, api_key=google_api_key, verbose=False)
    

    teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

    # Define the graph.
    super_builder = StateGraph(State)
    super_builder.add_node("supervisor", teams_supervisor_node)
    super_builder.add_node("research_team", call_research_team)
    super_builder.add_node("writing_team", call_paper_writing_team)

    super_builder.add_edge(START, "supervisor")
    super_graph = super_builder.compile()

    for s in super_graph.stream(
    {
        "messages": [
            ("user", question)
        ],
    },
    {"recursion_limit": 150},
):
        print(s)
        print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-question", help="The question to ask")
    args = parser.parse_args()
    main(args.question)


