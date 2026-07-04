from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests

from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import create_react_agent, AgentExecutor

from langchain import hub

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "agent-app"



# ---------------------------------------------------------
# DuckDuckGo Search Tool
# ---------------------------------------------------------

search_tool = DuckDuckGoSearchRun()


# ---------------------------------------------------------
# Custom Weather Tool
# ---------------------------------------------------------

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data
    for the given city.
    """

    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"

    response = requests.get(url)

    return response.json()


# ---------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------

llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0

)


# ---------------------------------------------------------
# Pull ReAct Prompt
# ---------------------------------------------------------

prompt = hub.pull("hwchase17/react")


# ---------------------------------------------------------
# Create ReAct Agent
# ---------------------------------------------------------

agent = create_react_agent(

    llm=llm,

    tools=[

        search_tool,

        get_weather_data

    ],

    prompt=prompt

)


# ---------------------------------------------------------
# Agent Executor
# ---------------------------------------------------------

agent_executor = AgentExecutor(

    agent=agent,

    tools=[

        search_tool,

        get_weather_data

    ],

    verbose=True,

    max_iterations=5

)


# ---------------------------------------------------------
# User Query
# ---------------------------------------------------------

response = agent_executor.invoke({

    "input": "What is the current temp of gurgaon"

})

print(response)

print(response["output"])