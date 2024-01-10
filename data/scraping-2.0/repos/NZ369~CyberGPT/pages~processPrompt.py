import os
import openai

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

from langchain import OpenAI, LLMChain
from langchain.memory import ConversationBufferMemory

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
import warnings
import requests
from langchain.chat_models import AzureChatOpenAI
from tools.abuseIPDB_tools import abuseIPDB_check_IP
from llms.azure_llms import create_llm

from dotenv import load_dotenv
#load environment variables
load_dotenv()

# Ignore warnings
warnings.filterwarnings('ignore')
# load environment variables

abuseIPDB_apikey = os.getenv("ABUSEIPDB_API_KEY")

def parse_text_from_response(response):
    if response:
        prompt = response.get('prompt')
        if prompt:
            text = prompt.get('text')
            if text:
                return text
    return None


def get_processPrompt(usecase):
    # Create this API key from Penfield Dashboard
    api_key = os.getenv("PENFIELD_PROCESSPROMPT_API_KEY")

    headers = {
        'x-api-key': api_key,
    }
    
    base_url = os.getenv("PENFIELD_PROCESSPROMPT_API_URL")
    url = f'{base_url}{usecase}'

    response = requests.get(url, headers=headers)

    # Check for successful request
    if response.status_code == 200:
        print("Success!")
        text = parse_text_from_response(response.json())
        return text  # This will return the response in JSON format
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None  # return None if the request failed

# ToDO: Create REST API, pass in Category Name. If Category Name Exists return processPrompt. If it doesn't exist create
processPrompt_output = ""

agent_memory = ConversationBufferMemory(memory_key="chat_history")

tools = [
    abuseIPDB_check_IP
]

# Prompt Template

# Set up the base template
template = """Answer the following questions as best you can, but speaking as professional cybersecurity consultant. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer. Follow these steps: {processPrompt_output}
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a professional cybersecurity consultant when giving your final answer.

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs['processPrompt_output'] = processPrompt_output
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

llm = create_llm(temp=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=agent_memory)

get_alert_category_prompt = PromptTemplate.from_template(
    "Your task is to extract Alert Title from a sentence. The alert title is always in quotes. An example of alert title extraction is as follows: Sentence: {prompt} .Alert Title:")

chainAlertCategory = LLMChain(llm=llm, prompt=get_alert_category_prompt)

langchain.debug = False

# UI
st.image("assets/penfield.png", width=200)

st.title('CyberGPT Virtual Analyst')
st.subheader(
    'Powered by ProcessPrompt')

prompt = st.text_area(
    "Hello! As your Virtual Cybersecurity Analyst, proficient with your processes and tools, I'm here to solve alerts. Please provide the alert title in double quotes along with any relevant details. Thanks!", height=3)
if "memory" not in st.session_state:
    st.session_state["memory"] = ""

if st.button("Analyze"):

    st.header("1. Alert Title Extraction")
    alertCategory = chainAlertCategory.run(prompt).replace('"', '').strip()
    st.markdown("First, I am extracting the Alert Title from your request.")
    with st.expander("Extracted Alert Title"):
        st.markdown("**{}** ".format(alertCategory))

    st.header("2. Understand Process To Follow From ProcessPrompt")

    st.markdown("Next, I'll access the Penfield ProcessPrompt API to comprehend your organization's specific processes and tools used for this alert.")
    st.markdown(
        "This is crucial to prevent AI Agents from making risky assumptions (aka Hallucination) or failing to solve problems.")

    with st.expander("How Penfield Generates Process Information"):
        st.markdown("* Penfield uses a lightweight browser extension to produce real-time documentation and extract precise processes, guaranteeing updated knowledge.")
        st.markdown(
            "* Penfield is built on Kubernetes - all data remains private your your private cloud.")
        st.markdown(
            "* Penfield works with any LLM such as Azure OpenAI, OpenAI, Cohere or smaller Open Source Models.")
        st.image('assets/ProcessPrompt.png')

    processPrompt_output = get_processPrompt(alertCategory.lower())
    if processPrompt_output is not None:
        st.markdown("Great, ProcessPrompt just made me aware of your Process.")

        with st.expander(f"Process to solve {alertCategory} Generated by ProcessPrompt"):
            processPrompt_output = processPrompt_output.replace('. ', '.\n\n')
            st.markdown(processPrompt_output)

        st.header("3. Executing ProcessPrompt Steps")

        st.markdown("Next, I will run this Process...")

        print(chainAlertCategory.run(prompt))

        st_callback = StreamlitCallbackHandler(st.container())
        run = agent_executor.run(input=prompt, callbacks=[st_callback])
        st.session_state["memory"] += agent_memory.buffer
        with st.expander('Logs:'):
            memory = st.session_state["memory"].replace('. ', '.\n\n')
            st.info(memory)

        st.header("4. Verdict")
        st.markdown(run)

    else:
        st.header("ProcesPrompt Not Generated For This Use Case")
        st.markdown(
            "ProcessPrompt not available for this use case. Kindly put alert category in quotes "". Review available ProcessPrompts on Penfield Dashboard.")
        with st.expander('Example Usecase:'):
            st.markdown(
                'Use Case 1: Investigating Brute Force Failed alert with AbuseIPDB.')
            st.info(
                'I am investigating a "Brute force failed" attempt attack from ip 12.0.2.45. Investigate it and explain how.')
            st.markdown(
                'Use Case 2: Investigating Malicious File Found alert with AssemblyLine.')
            st.info(
                'I am investigating a "malicious file found" Alert with the two files named excel_sheet.xls and gw.txt. Investigate it and explain how.')
