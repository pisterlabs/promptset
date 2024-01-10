import traceback
import config

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish, LLMResult
from common.utils import generate_table
# import pika
import re
import json
import traceback
#import config
#from bots.utils import encode_message, decode_message
from common.rabbit_comms import publish, publish_action, consume, publish_actions
#from bots.langchain_assistant import generate_commands


class RabbitHandler(BaseCallbackHandler):

    #message_channel = pika.BlockingConnection()


    def __init__(self, color: Optional[str] = None, ) -> None:
        """Initialize callback handler."""
        #self.message_channel = message_channel
        #self.color = color
    # def on_llm_start(
    #     self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    # ) -> Any:
    #     print(f"prompts: {prompts}")

    # def on_agent_action(
    #     self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    # ) -> Any:
    #     #print(f"On Agent Action: {action.log}")
    #     thought_pattern = r'Thought: (.*)'
    #     match = re.search(thought_pattern, action.log)
    #     if match:
    #         message = match.group(1)
    #         #print("Thought:", thought)
    #         #"""Run on agent action."""
    #         #message = encode_message(config.USER_ID,'prompt', message)
    #         #self.message_channel.basic_publish(exchange='',routing_key='notify',body=message)
    #         publish(f"Thought: {message}")
    #         #print_text(action.log, color=color if color else self.color)
    #     observation_pattern = r'Observation: (.*)'
    #     obs_match = re.search(observation_pattern, action.log)
    #     if obs_match:
    #         observation = obs_match.group(1)
    #         #print("Thought:", thought)
    #         #"""Run on agent action."""
    #         #message = encode_message(config.USER_ID,'prompt', observation)
    #         #self.message_channel.basic_publish(exchange='',routing_key='notify',body=message)
    #         publish(f"Observation: {message}")
    #         #print_text(action.log, color=color if color else self.color)
    

    # def on_tool_end(
    #     self,
    #     output: str,
    #     *,
    #     run_id: UUID,
    #     parent_run_id: Optional[UUID] = None,
    #     **kwargs: Any,
    # ) -> None:
    #     """Run when tool ends running."""
    #     #table = generate_table(output)
    #     #publish(f"{table}")

    # def on_tool_error(
    #     self,
    #     error: Union[Exception, KeyboardInterrupt],
    #     *,
    #     run_id: UUID,
    #     parent_run_id: Optional[UUID] = None,
    #     **kwargs: Any,
    # ) -> Any:
    #     """Run when tool errors."""
    #     print("tool_error: " + str(error))
        

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            print(f"on_agent_finish Callback {finish}")
            #print(f"on_agent_finish Callback {finish.log}")
            message = finish
            if message:
                #message = encode_message(config.USER_ID,'on_agent_finish', message)
                #self.message_channel.basic_publish(exchange='',routing_key='notify',body=message)
                print(f"Agent Finish: {message}")
        except Exception as e:
            traceback.print_exc()

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            print(f"on_chain_end Callback {outputs}")
            #response = outputs.get("output")
            #look for output

            #look for text
            if outputs:

                # Convert the string to dictionary
                #dict_str = eval(outputs)

                # Extract 'text' content
                text_content = outputs.get("text")
                output_content = outputs.get("output")

                if text_content:
                    # Extract text part without JSON string
                    text_without_json = text_content.split('```')[0].strip()
                    text_without_prefixes = text_without_json.replace('Thought: ','').replace('Action:', '').replace('\n','')
                    if config.VERBOSE:
                        """Turn off"""
                        #publish(f"VERBOSE: {text_without_prefixes}")
                
                if output_content:
                    publish(f"{output_content}")
                    #buttons = [("Creating an Email", f"Create an email with the following: {output_content}"),("Creating a Task", f"Create a task with the following: {output_content}"),("Creating a Meeting", f"Create a meeting to discuss the following: {output_content}")]
                    #buttons = generate_commands(output_content)
                    #feedback = f"""{output_content}. Is there anything else I can do?"""
                    #publish_actions(feedback, buttons)
        except Exception as e:
            traceback.print_exc()
            #return e

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        if error:
            #message = encode_message(config.USER_ID,'prompt', message)
            #self.message_channel.basic_publish(exchange='',routing_key='notify',body=message)
            print(f"Chain Error: {error}")
            return str(error)