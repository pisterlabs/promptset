import os
from abc import ABC
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain import WolframAlphaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import LLMResult, AgentAction, AgentFinish
from langchain.tools import WolframAlphaQueryRun, format_tool_to_openai_function
from langchain.utils import print_text

os.environ['OPENAI_API_KEY'] = 'sb-48ce6279f88e82c385dfc0a1d0feb964f4ea485874f9aeb9'
os.environ['openai_api_base'] = 'https://api.openai-sb.com/v1'

os.environ["WOLFRAM_ALPHA_APPID"] = "5V6ELP-UUPQLEAUXU"
openai_api_key = 'sb-48ce6279f88e82c385dfc0a1d0feb964f4ea485874f9aeb9'

wolfram = WolframAlphaAPIWrapper()


# print(wolfram.run("What is 2x+5 = -3x + 7?"))
# tools = load_tools(["wolfram-alpha"])


class MyCustomHandlerTwo(BaseCallbackHandler):
    def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            **kwargs: Any,
    ) -> None:
        """Do nothing."""
        print(f"on_tool_start{serialized}")
        pass

    def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        print("on_tool_error")
        pass



class MyCustomHandlerTwo11(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        print(f"on_new_token {token}")
        pass

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"on_llm_start (I'm the second handler!!) {serialized}")
        print("on_llm_start")


    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        print("on_llm_end")
        pass


    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        print("on_llm_error")
        pass

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        #print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")
        print(f"on_chain_start{class_name};;;{serialized}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        # print("\n\033[1m> Finished chain.\033[0m")
        print("on_chain_end")

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        print("on_chain_error")
        pass

    def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            **kwargs: Any,
    ) -> None:
        """Do nothing."""
        print(f"on_tool_start{serialized}")
        pass

    def on_agent_action(
            self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print(f"action111111111111:{action.tool}")
        print_text(action.log, color='red')

    def on_tool_end(
            self,
            output: str,
            color: Optional[str] = None,
            observation_prefix: Optional[str] = None,
            llm_prefix: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        print("on_tool_end")
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")

    def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        print("on_tool_error")
        pass

    def on_text(
            self,
            text: str,
            color: Optional[str] = None,
            end: str = "",
            **kwargs: Any,
    ) -> None:
       #print(f"text:{text}")
       print_text(text, color=color , end="\n") 

    def on_agent_finish(
            self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print(f"finish.log:{finish.log}")
        print_text(finish.log, color=color , end="\n")


handler = StdOutCallbackHandler()

query_run = WolframAlphaQueryRun(api_wrapper=wolfram,callbacks=[MyCustomHandlerTwo()],tags=['a-tag'])

tools = [query_run]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, )

agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True
                         )

#logfile = "output.log"
# callbacks=[MyCustomHandlerTwo()] What is 2x+5 = -3x + 7?
reply = agent.run(input="2 * 2 * 0.13 - 1.001? 如何计算,用中文回复" ,callbacks=[MyCustomHandlerTwo11()])
print("--------------------------------------------------------------")
print(reply)
#logger.info(reply)
