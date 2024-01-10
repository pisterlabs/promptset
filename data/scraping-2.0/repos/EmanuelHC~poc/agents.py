import dotenv 
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
import re
import os 
import langchain
from langchain.agents import initialize_agent, AgentType
from googleapiclient.discovery import build

from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from abc import ABC, abstractmethod
# Load the .env file
load_dotenv()

from my_gmail_toolkit import CustomGmailToolkit


import logging

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')



# Retrieve the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in the .env file.")


class SkillfulAssistantAgent(ABC):
    def __init__(self):
        self._tools = self.build_tools()
        self.tools = self.set_tools()
        self.prompt_template = self.set_prompt_template()
        self.llm = OpenAI(temperature=0)
        self.memory = self.set_memory()
        self.output_parser = CustomOutputParser()

    def build_tools(self):
            tools = {
                #general web search tool
                "Web Search": Tool(
                    #
                    name = "Web Search",
                    func =  DuckDuckGoSearchRun().run,
                    description="useful for when you need to answer questions about current events",
                ),
                "Search WebMD" : Tool(
                    name = "Search WebMD",
                    func= self._search_webmd,
                    description="useful for when you need to answer medical and pharmalogical questions"
                    )
            }
            return tools
    
    def _search_webmd(self, input):
        search = DuckDuckGoSearchRun()
        search_results = search.run(f"site:webmd.com {input}")
        return search_results
    
    @abstractmethod
    def set_tools(self):
        pass
    @abstractmethod
    def set_memory(self):
        pass
    @abstractmethod
    def set_prompt_template(self):
        pass
    @abstractmethod
    def run_agent(self, input_prompt: str) -> str:
        pass

class MedicalAssistantAgent(SkillfulAssistantAgent):
    def __init__(self):
        super().__init__()
    
    def run_agent(self, input_prompt: str) -> str:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        tool_names = [tool.name for tool in self.tools]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser= self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        # Set agent executor
        # Agent Executors take an agent and tools and use the agent to decide which tools to call and in what order.
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=self.tools,
                                                    verbose=True,
                                                    memory=self.memory)
        agent_output = agent_executor.run(input_prompt)
        self.process = agent_output 
        print("debugging agent output: ", agent_output)
        final_answer = agent_output.split("Final Answer:")[-1].strip()
        return final_answer

    def set_tools(self):
        
        return [self._tools['Web Search'], self._tools['Search WebMD']]
    
    def _search_webmd(self, input):
        search = DuckDuckGoSearchRun()
        search_results = search.run(f"site:webmd.com {input}")
        return search_results

    def set_prompt_template(self):
        template = """Answer the following questions as best you can, but speaking as compasionate medical professional. You have access to the following tools:

                        {tools}

                        Use the following format:

                        Question: the input question you must answer
                        Thought: you should always think about what to do
                        Action: the action to take, should be one of [{tool_names}]
                        Action Input: the input to the action
                        Observation: the result of the action
                        ... (this Thought/Action/Action Input/Observation can repeat N times)
                        Thought: I now know the final answer
                        Final Answer: the final answer to the original input question

                        Begin! Remember to answer as a compansionate medical professional when giving your final answer.
                        Previous conversation history:
                        {history}
                        Question: {input}
                        {agent_scratchpad}"""
        return CustomPromptTemplate(
            template=template,
            tools=self.tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps", "history"]
        )

    def set_memory(self):
        memory=ConversationBufferWindowMemory(k=5)
        return memory
    
    def get_summary(self):
        return self.memory

class CustomOutputParser(AgentOutputParser):
    '''
    The output parser is responsible for parsing the LLM output into AgentAction and AgentFinish. This usually depends heavily on the prompt used.
    This is where you can change the parsing to do retries, handle whitespace, etc
    '''
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
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

class CustomPromptTemplate(StringPromptTemplate):
    '''
    This instructs the agent on what to do. Generally, the template should incorporate:

    tools: which tools the agent has access and how and when to call them.

    intermediate_steps: These are tuples of previous (AgentAction, Observation) pairs. These are generally not passed directly to the model, but the prompt template formats them in a specific way.

    input: generic user input
    '''
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



class GeneralAssistanAgent(SkillfulAssistantAgent):
    def __init__(self):
        self.set_gmail_tools()
        super().__init__()
      

    def set_gmail_tools(self):
        '''
        credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
        )
        api_resource =  build("gmail", 'v1', credentials=credentials)
        logging.debug(f'credentials is {credentials}')
        logging.debug(f'api resource is {api_resource}')
        print(f'debug', credentials)
        print('debug',api_resource)
        #api_resource = build_resource_service(credentials=credentials)
        #toolkit = GmailToolkit(api_resource=api_resource) 
        #GmailToolkit.update_forward_refs()
        #toolkit = GmailToolkit(api_resource=api_resource) 
        
        toolkit = CustomGmailToolkit()
        GmailToolkit.update_forward_refs()
        '''
        toolkit = GmailToolkit()
        self.gmail_tools = toolkit.get_tools()

    def set_tools(self):
        return self.gmail_tools
    def set_memory(self):
        return ''
    def set_prompt_template(self):
        return ''
    def run_agent(self, input_prompt: str) -> str:
        llm = OpenAI(temperature=0)
        agent = initialize_agent(
        tools= self.tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
        return agent.run(input_prompt)
