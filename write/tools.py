from pathlib import Path
import os
from typing import Dict, Optional, Annotated, List

from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict
from langchain_core.tools import tool


WORKING_DIRECTORY = Path("OUTPUT")
if not WORKING_DIRECTORY.exists():
    WORKING_DIRECTORY.mkdir()

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    
    """
    Create an outline from a list of main points or sections and save it to a text file.

    Args:
        points (List[str]): List of main points or sections.
        file_name (str): File path to save the outline.

    Returns:
        str: Path of the saved outline file.
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """
    Read a document from a text file and return its content as a string.

    Args:
        file_name (str): File path to read the document from.
        start (int, optional): The start line. Defaults to 0.
        end (int, optional): The end line. Defaults to None.

    Returns:
        str: The content of the document between the start and end lines.
    """

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:

    """
    Write a document from a string content and save it to a text file.

    Args:
        content (str): Text content to be written into the document.
        file_name (str): File path to save the document.

    Returns:
        str: Path of the saved document file.
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:

    """
    Edit a document by inserting text at specific line numbers.

    Args:
        file_name (str): Path of the document to be edited.
        inserts (Dict[int, str]): Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.

    Returns:
        str: Path of the edited document file.
    """
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"



repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """
    Execute a python code snippet to generate a chart and return the stdout of the execution.

    Args:
        code (str): The python code to execute to generate your chart.

    Returns:
        str: The result of the execution. If the execution failed, the error message is returned.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"