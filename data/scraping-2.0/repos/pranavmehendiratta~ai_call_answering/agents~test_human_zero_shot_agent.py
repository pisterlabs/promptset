from uuid import UUID
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate, PromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.memory import ConversationBufferMemory, CombinedMemory
from langchain.chat_models import ChatOpenAI
from typing import Any, Dict, List, Optional, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from test_human_system_prompt import test_human_system_prompt
from test_human_human_prompt import test_human_human_prompt
import langchain
from role_playing_zero_shot_agent import assistant
import role_playing_zero_shot_agent
import ast
import os
from common.utils import SCRATCH_SPACE_DIR_PATH
from langchain.callbacks.base import BaseCallbackHandler
import json


test_human_system_message_prompt = SystemMessagePromptTemplate(prompt=test_human_system_prompt)
test_human_human_message_prompt = HumanMessagePromptTemplate(prompt=test_human_human_prompt)

AGENT_DIR_PREFIX = "test_human"
AGENT_DIR_PATH = f"{SCRATCH_SPACE_DIR_PATH}/{AGENT_DIR_PREFIX}"
os.mkdir(AGENT_DIR_PATH)

_chat_file = open(f"{AGENT_DIR_PATH}/chat.txt", "w")

STOP_TOKENS = ["\nMe:"]

class TestOnToolCallbackHandler(BaseCallbackHandler):
    global _chat_file
    _chat_file.write(f"{test_human_human_prompt.format(intermediate_steps = '')}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:  
        result = super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
        #_chat_file.write("{inputs}")
        return result

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        result = super().on_tool_start(serialized, input_str, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
        #print(f"test_human on_tool_start input_str = {input_str}")
        return result

    def on_tool_end(self, output: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        result = super().on_tool_end(output, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        #print(f"test_human on_tool_end output = {output}")
        _chat_file.write(f"\nMe: {output}\nYour Response: ")
        return result
    
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        result = super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        #print(f"test_human on_chain_end outputs = {outputs}")
        if 'output' in outputs:
            _chat_file.write(f"{outputs['output']}")
        elif 'text' in outputs:
            _chat_file.write(f"{outputs['text']}")
        return result


class TestHumanAgentOutputParser(AgentOutputParser):
    global _chat_file

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        #print(llm_output)
        if "[task_end]" in llm_output:
            #print("Ending human conversation")
            #parsed_output_match = re.search(r"\s*Human: \[end\]\s*(?=\n|$)", llm_output)
            #parsed_output = parsed_output_match.group(1) if parsed_output_match else None
            #print(f"parsed_output = {parsed_output}")
            output = llm_output.replace("[task_end]", "")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output":output},
                log=llm_output,
            )        

        # Parse out the Function and Function input
        human_match = re.search(r"\s*(.*?)(?=\n|$)", llm_output)
        human_message = human_match.group(1) if human_match else None

        #print(f"[Your Response]: {human_message}") 

        if human_message is None:
            raise ValueError("Human message is None")

        # Extract the argument
        human_message = human_message.strip() 

        # input to the assistant tool
        tool_input = {"question": human_message}

        #_chat_file.write(f"{human_message}\n")

        # Return the action and action input
        return AgentAction(tool="assistant", tool_input=tool_input, log=llm_output)


output_parser = TestHumanAgentOutputParser()
history = [test_human_system_message_prompt, test_human_human_message_prompt]
llm = ChatOpenAI(temperature=0.7, model="gpt-4")
chat_prompt = ChatPromptTemplate.from_messages(history)
llm_chain = LLMChain(
    llm=llm, 
    prompt=chat_prompt,
    custom_color = "red"
)
tools = [assistant]
tool_names = [tool.name for tool in tools]

test_human_agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=STOP_TOKENS,
    allowed_tools=tool_names
)

test_human_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=test_human_agent, 
    tools=tools,
    #verbose=True,
    #max_iterations=2
)