import os
from dotenv import load_dotenv
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_core.tools import tool

import prompts

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    model = "gpt-oss-20b",
    base_url=BASE_URL,
    api_key=OPENROUTER_API_KEY,
    temperature=1,
    default_headers={
        "HTTP-Referer": "http://localhost",   
        "X-Title": "Agent for project"
    }
)

tavily = TavilyClient()

# print(model.invoke("What is AI").content)
@tool
def web_search(query: str, 
               max_result : int = 3, 
               topic : Literal["general", "news", "finance"] = "general"):
    """Search web for content"""
    
    return tavily.search(query=query, max_results=max_result, topic=topic)
    
agent = create_deep_agent(
    tools=[web_search],
    instructions=prompts.SYSTEM_PROMPT,
    model=model
)

result = agent.invoke({"messages" : [{
    "role" : "user",
    "content" : "What is langgraph ?"
    }]})

print(result['messages'])