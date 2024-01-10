from typing import Any, Dict, List, Tuple, Optional, Union

from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.types import AgentType
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain.tools.base import BaseTool
from langchain.agents.agent import Agent

from langchain.agents.agent_toolkits.pandas.base import _get_prompt_and_tools, _get_single_prompt

import logging

class TOTVSAgentExecutor(AgentExecutor):
    pass
#     def _return(
#         self,
#         output: AgentFinish,
#         intermediate_steps: list,
#         run_manager: Optional[CallbackManagerForChainRun] = None,
#     ) -> Dict[str, Any]:
#         print(">>> _return:: ")
#         print(output)
#         parentReturn = super()._return(output, intermediate_steps, run_manager)
#         return parentReturn
    
#     def _take_next_step(
#         self,
#         name_to_tool_map: Dict[str, BaseTool],
#         color_mapping: Dict[str, str],
#         inputs: Dict[str, str],
#         intermediate_steps: List[Tuple[AgentAction, str]],
#         run_manager: Optional[CallbackManagerForChainRun] = None,
#     ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
#         parentReturn = super()._take_next_step(name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager)
#         print(">>> _take_next_step :: ")
#         print(parentReturn)

#         for item in parentReturn:
#             print(item)
#             print(type(item))

#         return parentReturn

class TOTVSAgent(ZeroShotAgent):
    pass
    # def plan(
    #     self,
    #     intermediate_steps: List[Tuple[AgentAction, str]],
    #     callbacks: Callbacks = None,
    #     **kwargs: Any,
    # ) -> Union[AgentAction, AgentFinish]:
    #     parentReturn = super().plan(intermediate_steps, callbacks, **kwargs)

    #     # if(isinstance(parentReturn, AgentAction)):
    #     #     print(" tool_input :::: ")
    #     #     print(parentReturn.tool_input)
    #         # parentReturn.tool_input = "import geopandas\nimport geopandas as gpd\nimport geopy\nimport geopy as gpy\nimport pandas as pd\nfrom geopy.distance import geodesic\nfrom shapely.geometry import Point\nfrom shapely import wkt\nworld = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))\n" + parentReturn.tool_input

    #     return parentReturn

def create_TOTVS_agent(
    llm: BaseLanguageModel,
    df: Any,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    agent: BaseSingleActionAgent
    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt, tools = _get_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = TOTVSAgent( #ZeroShotAgent
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            callback_manager=callback_manager,
            **kwargs,
        )
    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        _prompt, tools = _get_functions_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
        )
        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")
    return TOTVSAgentExecutor.from_agent_and_tools( #AgentExecutor #TOTVSAgentExecutor
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )


"""Callback Handler that prints to std out."""
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish, LLMResult


class TOTVSCallback(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        logging.warning("on_llm_start:" + str(prompts))
        logging.warning(" ")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        logging.warning("on_llm_end:" + str(response))
        logging.warning("")
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        logging.warning("on_llm_new_token:" + str(str))
        logging.warning(" ")
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        logging.warning("on_llm_error:")
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", "")
        logging.warning(f"\n\n\033[1m> EEntering new {class_name} chain...\033[0m")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\n\033[1m> FFinished chain.\033[0m")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print_text(action.log, color=color if color else self.color)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color if color else self.color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        print_text(text, color=color if color else self.color, end=end)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print_text(finish.log, color=color if self.color else color, end="\n")