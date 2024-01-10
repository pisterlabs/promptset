"""
langchain==0.0.276
langchain-experimental==0.0.11

LangChain PlanAndExecute: https://python.langchain.com/docs/modules/agents/agent_types/plan_and_execute

This file uses streamlit, so `streamlit run agents_example.py` to run it.

Run `python3 weather_app.py` to start the required weather service.
Run `python3 ./store_and_stock_app.py` to start the required store and stock services.

- [2305.04091] Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models: https://arxiv.org/abs/2305.04091

- [2303.11366] Reflexion: Language Agents with Verbal Reinforcement Learning: https://arxiv.org/abs/2303.11366

- SmartGPT: Major Benchmark Broken - 89.0% on MMLU + Exam's Many Errors https://youtu.be/hVade_8H8mE
"""

import streamlit as st
from termcolor import colored
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from common import get_llm
from dotenv import find_dotenv, load_dotenv
import json

from catalog import get_catalog

usePlanAndExecuteAgentType = True
useBuiltInSearchAndCalculatorTools = False
useUserInputTool = True

if 'user_input_history' not in st.session_state:
    st.session_state['user_input_history'] = []

user_input_history = st.session_state['user_input_history']

st.set_page_config(page_title="🥼🧪 Experimenting With Langchain Agents And \"Knowledge Provider Mesh\"")
st.title("🥼🧪 Experimenting With Langchain Agents And \"Knowledge Provider Mesh\"")

if usePlanAndExecuteAgentType:
    st.info("Hi. This agent is using PlanAndExecute agent type. Type your query and sit submit.")
else:
    st.info("Hi. This agent is using STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION agent type. Type your query and sit submit.")


def get_user_input(query: str) -> str:
    # print and query on console and get user input
    print(colored("\n\nEntering User Input Handler\n\n", "green", "on_white", attrs=["bold"]))
    print(f"Agent: {query}")

    print(colored("Please enter your response: ", "cyan"))
    user_input = input()

    user_input_history.append({
        "agent_query": query,
        "user_response": user_input
    })

    return user_input

@st.cache_resource()
def setup_agent():
    load_dotenv(find_dotenv())

    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = get_llm()

    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = []

    # Built in tools and functions
    if useBuiltInSearchAndCalculatorTools:
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events",
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math",
            )
        ]

    if useUserInputTool and usePlanAndExecuteAgentType:
        # add user input tool for plan and execute agent as it doesn't support memory yet
        tools.append(
            Tool(name="UserInput",
                func=get_user_input,
                description="""
                Useful for when you need to get further clarification from the user if you're unsure how to answer their question.
                Important: Only useful when you can't infer the user's intent from their question and need an explicit answer.
                If you can infer the answer, then you don't need to use this tool. Don't bother the user with unnecessary questions.

                Input should be in the following JSON format. i.e. {{\"query\":\"What type of drill are you interested in?\"}}

                When presenting or summarising this interaction, always present it like...
                AI agent asked: <the question>
                User answered: <the answer>

                This will be helpful for subsequent steps.
                """
            )
        )

    # add knowledge provider tools
    knowledge_tools = [service.get_tool() for service in get_catalog()]
    tools.extend(knowledge_tools)

    if usePlanAndExecuteAgentType:
        # plan and execute - https://python.langchain.com/docs/modules/agents/agent_types/plan_and_execute
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
    else :
        # structured tool - https://python.langchain.com/docs/modules/agents/agent_types/structured_chat.html
        agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                memory=memory,
                                agent_kwargs={
                                    "memory_prompts": [chat_history],
                                    "input_variables": ["input", "agent_scratchpad", "chat_history"]
                                },
                                verbose=True)

    return agent

agent = setup_agent()

def generate_response(input_text):

    with st.spinner(text="Generating... Please check the agent backend to see if it requires further user input."):
        response = agent({"input": input_text})

        st.info(response["output"], icon="🤖")

        st.divider()
        st.caption("Additional User Input Used During Run")
        user_input_used = json.dumps(user_input_history)
        st.json(user_input_used, expanded=True)
        st.session_state['user_input_history'] = []


### search and math tools
# Who was Leo DiCaprio's girlfriend in 2021? What is her age in 2023 divided by 2? What's the weather of the city she was born in as of 28/08/2023?

### Weather tool
# What is the weather forecast for Melbourne on 28/08/2023?

### Hardys stock knowledge provider
# What are the 3 closest Hardy stores to Heathmont with the Ryobi One Plus 18V Drill and the qty available?
# What are the 3 closest Hardy stores to Heathmont with the Ryobi One Plus 18V Drill currently in stock today and their qty available? Show them in a markdown table.
# I'm looking to purchase the "Ryobi One Plus 18V Drill" today. I need to find the 2 Hardy stores nearest to Heathmont that have it on stock.

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the 3 closest Hardy stores to Heathmont with the Ryobi One Plus 18V Drill and the qty available?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
