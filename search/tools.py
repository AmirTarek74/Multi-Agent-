from typing import Annotated, List
from dotenv import load_dotenv
import os 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
load_dotenv()  
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(max_results=5)


@tool
def scrape_webpages(urls: List[str]) -> str:

    """
    Scrape a list of webpages and return their content in a single string, with 
    each webpage separated by two newlines. Each webpage is also wrapped in a 
    <Document> tag with the title of the webpage in the name attribute.

    Args:
        urls (List[str]): The URLs to scrape.

    Returns:
        str: The scraped content of the webpages.
    """
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )