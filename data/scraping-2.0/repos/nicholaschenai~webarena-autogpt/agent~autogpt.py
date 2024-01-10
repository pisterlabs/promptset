"""
adapted from https://github.com/younghuman/LLMAgent
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from .autogpt_output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
# from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
# from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
#     FINISH_NAME,
# )
# from langchain_experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from .autogpt_prompt import AutoGPTPrompt

from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
# from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever
# from expert_action_predictor import ExpertActionPredictor
# from autogpt_output_parser import AutoGPTAction
# import numpy as np
# import json
# import torch
# import re

DELIMITER = "%%%%"


class AutoGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        # feedback_tool: Optional[HumanInputRun] = None,
        loop_limit: int = 100,
        init_obs: str = None,
        # expert_predictor: ExpertActionPredictor = None,
        top_k: int = 1,
        random: bool = False
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        # self.feedback_tool = feedback_tool
        self.loop_limit = loop_limit
        self.init_obs = init_obs
        # self.expert_predictor = expert_predictor
        self.top_k = top_k
        self.random = random


    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
            send_token_limit: int = 4097,
            max_tokens: int = 1000,
            base_plus_mem_tokens: int = 2500
    ) -> AutoGPT:
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
            max_tokens=max_tokens,
            send_token_limit=send_token_limit,
            base_plus_mem_tokens=base_plus_mem_tokens
        )
        # human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            # feedback_tool=human_feedback_tool,
        )

    def run(self, goals: List[str]) -> str:
        # Interaction Loop
        loop_count = 0
        actions = []
        cur_obs, cur_info, result = None, None, None

        if self.init_obs:
            # cur_obs, cur_info = self.init_obs.split(DELIMITER)
            cur_obs = self.init_obs
        while loop_count < self.loop_limit:
            user_input = (
                "Determine which next command to use, "
                "and respond using the JSON format specified above:"
            )
            # Discontinue if continuous limit is reached
            loop_count += 1
            loop_msg = f"loop number:{loop_count}"
            self.full_message_history.append(SystemMessage(content=loop_msg))
            if cur_obs and loop_count == 1:
                self.full_message_history.append(SystemMessage(content=cur_obs))
                print(cur_obs)
            elif result:
                pass
                #print(result)

            # if cur_obs and cur_info and self.expert_predictor:
            #     info = json.loads(cur_info)
            #     ## Dummy controller for ALFWorld Observation String
            #     if 'alfworld_ob' in info and info['alfworld_ob'] is not None:
            #         alfworld_ob = info.pop('alfworld_ob')
            #         action = self.expert_predictor.predict(alfworld_ob, info)
            #         user_input = f"Here's one suggestion for the command: {action}.\n" +\
            #                     "Please use this suggestion as a reference and make your own judgement. " + user_input
            #     else:
            #         if 'image_feat' in info and info['image_feat'] is not None:
            #             info['image_feat'] = torch.tensor(info['image_feat'])
            #         cur_obs = cur_obs.replace("=Observation=\n", "")
            #         # print("########", cur_obs, info)
            #
            #         action = self.expert_predictor.predict(cur_obs, info, top_k=self.top_k, random=self.random)
            #         if self.top_k == 1:
            #             tool_name, tool_input = action.replace("]", "").split("[")
            #             action = f"{tool_name} with '{tool_input}'"
            #             user_input = f"Here's one suggestion for the command: {action}.\n" +\
            #                         "Please use this suggestion as a reference and make your own judgement. " + user_input
            #         else:
            #             action_list = []
            #             for a in action:
            #                 tool_name, tool_input = a.replace("]", "").split("[")
            #                 action_list.append(f"{tool_name} with '{tool_input}'")
            #             action = ", ".join(action_list)
            #             user_input = f"Here's a few suggestions for the command: {action}.\n" +\
            #                         "Please use this suggestion as a reference and make your own judgement. " + user_input
            
            print(loop_msg)
            print(user_input)
            # Send message to AI, get response
            try:
                assistant_reply = self.chain.run(
                    goals=goals,
                    messages=self.full_message_history,
                    memory=self.memory,
                    user_input=user_input,
                )
            except Exception as e:
                print(f"Error: {str(e)}, {type(e).__name__}")
                raise e


            # Print Assistant thoughts
            print(assistant_reply)
            self.full_message_history.append(HumanMessage(content=user_input))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)

            # if result:
            #     match = re.search(r"\{'reward':([\d\.]+)\}", result)
            #     if match:
            #         action = AutoGPTAction(
            #             name="finish",
            #             args={"response": "I have successfully purchased the hair mask and completed all my objectives."}
            #         )
            actions.append(action)
            cur_obs, cur_info = None, None
            tools = {t.name: t for t in self.tools}
            # if action.name == FINISH_NAME:
            #     break
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                # results = observation.split(DELIMITER)
                # if len(results) == 2:
                #     cur_obs, cur_info = results
                # else:
                #     cur_obs = results[0]
                cur_obs = observation
                result = f"Command {tool.name} returned: {cur_obs}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )
            if tool.name == 'stop':
                break
            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            # if self.feedback_tool is not None:
            #     feedback = f"\n{self.feedback_tool.run('Input: ')}"
            #     if feedback in {"q", "stop"}:
            #         print("EXITING")
            #         break
            #     memory_to_add += feedback
            print(result)
            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))
        return self.full_message_history, actions