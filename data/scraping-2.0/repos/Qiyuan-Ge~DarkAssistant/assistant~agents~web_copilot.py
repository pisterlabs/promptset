import streamlit as st
from typing import List, Union
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.tools.base import BaseTool
from langchain.prompts import StringPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from fastchat.conversation import get_conv_template

from .helper import get_current_time, get_thoughts_from_intermediate_steps
from .parser import OutputParserForAgent


SYSTEM_MESSAGE = """\
You are an AI assistant that helps people find information. You should utilize a variety of tools to collect information and generate a comprehensive, reliable answer."""


SPLIT_TOKEN = '<end of turn>'


prompt_template = """
Current datetime: {date}


You have access to the following tools:
{tools}


Question: the input question you must answer

You should only respond in format as described below:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: {{"arg name": "value"}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated one or more times)


Here is an example:

Question: Who create Spiderman?
Thought: I need to gather information about who created Spiderman.
Action: Wikipedia with question
Action Input: {{"input": "Spiderman", "question": "Who created Spiderman?"}}
Observation: Stan Lee and Steve Ditko created Spider-Man.
Thought: I know who created Spiderman now.
Action: Final Response
Action Input: Stan Lee and Steve Ditko created Spider-Man.


Now let's answer the following question:

Question: {question}""" + SPLIT_TOKEN + "{agent_scratchpad}"


class ConvPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Union[Tool, BaseTool]]

    def format(self, **kwargs) -> str:
        # intermediate steps(Action, Observation)
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = get_thoughts_from_intermediate_steps(intermediate_steps)
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{i+1}. {tool.name}: {tool.description}" for i, tool in enumerate(self.tools)])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["date"] = get_current_time()
        
        agent_profile = kwargs.pop("agent_profile")
        template_name = kwargs.pop("template_name")
        conv = get_conv_template(template_name)
        conv.system_message = agent_profile
        conv.sep2 = ''
        
        prompt = self.template.format(**kwargs)
        user_prompt, assistant_prompt = prompt.split(SPLIT_TOKEN)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], assistant_prompt)

        return conv.get_prompt()


class WebCopilot:
    def __init__(
        self,
        tools: List[Union[Tool, BaseTool]], 
        model_name: str = "text-davinci-003",  
        generate_params: dict = {'max_tokens':2048, 'temperature':0.9, 'top_p':0.6},
        agent_profile: str = SYSTEM_MESSAGE,
        template_name: str = "vicuna_v1.1",
        callbacks = None,
        ):
        self.name = "Web Copilot"
        self.agent_profile = agent_profile
        self.template_name = template_name
        self.llm = OpenAI(
            model=model_name,
            temperature=generate_params['temperature'],
            max_tokens=generate_params['max_tokens'],
            top_p=generate_params['top_p'],
            streaming=True, 
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.prompt_template = ConvPromptTemplate(
            template=prompt_template,
            tools=tools,
            input_variables=["question", "intermediate_steps", "agent_profile", "template_name"]
        )
        self.agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=self.prompt_template),
            output_parser=OutputParserForAgent(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in tools],
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=tools, verbose=False)
        self.callbacks = callbacks
        
    def run(self, question: str):
        """_summary_

        Args:
            question (str): _description_
        """  
        st.toast(f'Assign the task to {self.name}', icon='🕵️‍♂️')
        try:
            response = self.agent_executor.run(
                {"question": question, "agent_profile": self.agent_profile, "template_name": self.template_name},
                callbacks=self.callbacks,
            )
        except Exception as e:
            response = f"Error: {e}"
        st.toast(f'{self.name} complete the task', icon='🎉')
        return response
    