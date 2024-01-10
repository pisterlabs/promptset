from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, tool
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import List, Union
from bs4 import BeautifulSoup
import requests
import re

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
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": "There was an error parsing the output. Please try again."},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class ChatWithTool():
    def __init__(self, tools):
        self.required_init = True
        self.tools = tools

    def init_agent(self, model_name, background, human_role, temperature, memory_size):
        prompt_with_history = CustomPromptTemplate(
            template=self.create_template(background, human_role),
            tools=self.tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps", "history"]
        )

        llm=ChatOpenAI(model_name=model_name, temperature=temperature)

        output_parser = CustomOutputParser()
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

        tool_names = [tool.name for tool in self.tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        memory=ConversationBufferWindowMemory(k=memory_size, ai_prefix="You: ", human_prefix=f"{human_role.capitalize()}: ")
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, verbose=True, memory=memory)
        self.required_init = False

    def create_template(self, background: str, human_role: str) -> str:
        template = f"""This is your background information:

        {background}""" + f"""

        Now you are here to answer your {human_role} questions as best you can.""" + """

        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]. if you can answer without tool you can skip and go straight to Final Answer.
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question it's very important to give your final answer in this format

        Begin! Remember that if you can answer without tool you can skip and go straight to "Final Answer: " without any Action/Action Input/Observation and always wrap your final answer in "Final Answer: "

        Previous conversation history:
        {history}

        New question: {input}
        {agent_scratchpad}"""
        return template
    
    def set_required_init(self, required_init):
        self.required_init = required_init

@tool
def execute_code(code) -> str:
    """Executes the code and returns the output"""
    try:
        return eval(code)
    except Exception as e:
        return str(e)

@tool
def read_content_from_url(url) -> str:
    """reads the content from the url"""
    try:
        url = url[url.index("http"):]
    except:
        return "Error: Invalid URL"
    try:
        # Send a GET request to fetch the content of the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the web content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the main content and remove HTML tags
            content = None
            if soup.find("main"):
                content = soup.find("main").text
            else:
                content = soup.text
            # Strip leading and trailing \n and spaces
            content = content.strip("\n").strip(" ")
            # Remove extra newlines
            content = re.sub(r"\n+", "\n", content)
            # Remove extra spaces
            content = re.sub(r"\s+", " ", content)
            # Remove extra tabs
            content = re.sub(r"\t+", "\t", content)
            # Cut into first 13000 characters (around 3250 tokens)
            content = content[:min(13000, len(content))]
            return content
        else:
            return f"Error: Unable to fetch content from url. Status code: {response.status_code}"

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to fetch content from url"