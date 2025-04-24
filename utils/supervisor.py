from typing import List, Optional, Literal, TypedDict
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages


class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    """
    Creates a supervisor node that routes to one of the given workers
    after consulting an LLM. The LLM is given a system prompt that
    describes the workers and the task, as well as the user's request and
    the workers' responses. The LLM then responds with the worker to act
    next. If no workers are needed, the LLM should respond with FINISH.

    Args:
        llm (BaseChatModel): The LLM to use for routing.
        members (list[str]): The workers to route to.

    Returns:
        str: The name of the supervisor node.
    """

    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:

        """"
        Consults an LLM to determine which worker to route to next, given
        the user's request and the workers' responses. If no workers are
        needed, the LLM should respond with FINISH.

        Args:
            state (State): The state of the graph.

        Returns:
            Command[Literal[*members, "__end__"]]: The command to route to
            the next worker, or to end the conversation if no workers are
            needed.
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node