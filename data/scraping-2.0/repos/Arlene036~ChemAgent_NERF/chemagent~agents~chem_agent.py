from typing import Any, Dict, List, Optional
from .few_shot_agent import FewShotAgent
from .few_shot_agent import FewShotAgentExecutor
from langchain import LLMChain
from langchain.tools.base import BaseTool
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .prompts import *
import nest_asyncio
from .tools import *
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import langchain

def _make_llm(model, temp, verbose):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm

class ChemAgent:
    def __init__(self,
                 tools: Sequence[BaseTool] = None,
                 model="gpt-3.5-turbo-0613",
                 analysis_model = "gpt-3.5-turbo-0613",
                 temp=0.1,
                 max_iterations=40,
                 verbose=True):
        if tools is None:
            tools = make_tools(model, verbose=verbose)


        self.llm_lists = [_make_llm(model, temp, verbose=verbose),
                          _make_llm(analysis_model, temp, verbose=verbose)]
        agent = FewShotAgent.from_llm_and_tools( # FewShotAgent里面包含多个LLMChain
            llms=self.llm_lists,
            tools=tools,
        ) 



        self.agent_executor = FewShotAgentExecutor.from_agent_and_tools(
            agent = agent,
            tools = tools,
            verbose = verbose,
            max_iterations = max_iterations,
            return_intermediate_steps = True
        )
    
    nest_asyncio.apply()  # Fix "this event loop is already running" error


    def run(self, init_prompt):
        outputs = self.agent_executor({"input": init_prompt}) # agent_executor是langchain Chain类的一个子类，有__call__方法
        # Parse long output (with intermediate steps)
        intermed = outputs["intermediate_steps"]

        final = ""
        for step in intermed:
            final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
        final += f"Final Answer: {outputs['output']}"
        
        final_answer = outputs['output']
        return final_answer, final # final: 过程
    
