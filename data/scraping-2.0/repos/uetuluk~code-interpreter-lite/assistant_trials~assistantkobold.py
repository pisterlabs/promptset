import gradio as gr
from dotenv import load_dotenv
import os
import requests
import io
import re
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import TextGen
from langchain.chat_models import ChatAnthropic
from langchain.agents import ZeroShotAgent, initialize_agent, AgentType, AgentExecutor
from langchain.tools import StructuredTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, LLMResult, AgentFinish, OutputParserException
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_toolkits import create_python_agent
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from typing import Any, Dict, List
from langchain.llms.human import HumanInputLLM


def setup_assistant():

    # class CustomLLM(LLM):
    #     n: int

    #     @property
    #     def _llm_type(self) -> str:
    #         return "custom"

    #     def _call(
    #         self,
    #         prompt: str,
    #         stop: Optional[List[str]] = None,
    #         run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     ) -> str:
    #         if stop is not None:
    #             raise ValueError("stop kwargs are not permitted.")
    #         return prompt[: self.n]

    #     @property
    #     def _identifying_params(self) -> Mapping[str, Any]:
    #         """Get the identifying parameters."""
    #         return {"n": self.n}

    # llm = HumanInputLLM(
    #     prompt_func=lambda prompt: print(
    #         f"\n===PROMPT====\n{prompt}\n=====END OF PROMPT======"
    #     )
    # )

    # tools = [PythonREPLTool()]

    # memory = ConversationBufferMemory(
    #     memory_key="chat_history")

    # agent_executor_instance = create_python_agent(
    #     llm=custom_llm,
    #     tool=PythonREPLTool(),
    #     verbose=True,
    #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # )

    # agent_executor_instance = initialize_agent(
    #     tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    from langchain.llms import KoboldApiLLM
    from langchain.agents import XMLAgent, tool, AgentExecutor

    @tool
    def search(query: str) -> str:
        """Search things about current events."""
        return "32 degrees"

    llm = KoboldApiLLM(endpoint="http://localhost:5001", max_length=80)

    chain = LLMChain(
        llm=llm,
        prompt=XMLAgent.get_default_prompt(),
        output_parser=XMLAgent.get_default_output_parser()
    )

    agent = XMLAgent(tools=[search], llm_chain=chain)

    agent_executor_instance = AgentExecutor(
        agent=agent, tools=[search], verbose=True)

    return agent_executor_instance


agent_executor = setup_assistant()

with gr.Blocks() as demo:

    with gr.Column() as chatbot_column:
        chatbot = gr.Chatbot()
        with gr.Row() as chatbot_input:
            msg = gr.Textbox(placeholder="Type your message here", lines=8)
            send = gr.Button(value="Send", variant="primary")

    def chatbot_handle(chatbot_instance, msg_instance):

        class ChatbotHandler(BaseCallbackHandler):
            def __init__(self):
                self.chatbot_response = ""
                super().__init__()

            def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
                self.chatbot_response += outputs.get("output", "") + '\n'

            def on_tool_end(self, output: str, **kwargs: Any) -> Any:
                self.chatbot_response += f'```\n{output}\n```\n'

            def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
                chatbot_thought = action.log.split("\n")[0]
                chatbot_thought = chatbot_thought.replace("Thought: ", "")

                if isinstance(action.tool_input, str):
                    chatbot_tool_input_code_string = action.tool_input
                else:
                    chatbot_tool_input_code_string = action.tool_input.get(
                        "code")
                self.chatbot_response += f"{chatbot_thought}\n"
                self.chatbot_response += f'```\n{chatbot_tool_input_code_string}\n```\n'

            def get_chatbot_response(self):
                return self.chatbot_response

        try:
            chatbotHandler = ChatbotHandler()
            agent_executor(
                msg_instance, callbacks=[chatbotHandler])
            chatbot_response = chatbotHandler.get_chatbot_response()

        except OutputParserException as e:
            raise gr.Error(
                "Assistant could not handle the request. Error: " + str(e))

        chatbot_instance.append((msg_instance, chatbot_response))

        return {
            chatbot: chatbot_instance,
            msg: "",
        }

    send.click(chatbot_handle, [chatbot, msg], [
        chatbot, msg])


if __name__ == "__main__":
    demo.launch(server_port=7860)
