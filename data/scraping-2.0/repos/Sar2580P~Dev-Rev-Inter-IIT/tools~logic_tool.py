from langchain.tools import BaseTool
from typing import Optional, List, Any 
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import sys, os
sys.path.append(os.getcwd())
from utils.llm_utility import llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils.templates_prompts import LOGICAL_TEMPLATE
import ast


generate_code_prompt = PromptTemplate(template=LOGICAL_TEMPLATE, input_variables=['query' , 'language'])
generate_code = LLMChain(llm = llm , prompt=generate_code_prompt)

class LogicalTool(BaseTool):
    name = "logic_tool"
    description = '''
    -  Use this tool, for various logical operations like conditional statements, while loops, addition, subtraction, iterate over lists etc. 
    - The input to this tool is symbolic and the output is a pseudocode to execute the task on input.
    - By symbolic, it means that it is in '$$PREV[i]' format, so, llm can't perform logic on it.
    '''
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        print('\ninside logic_tool tool...')
        code = generate_code.run({'query' : query , 'language' : 'python'})
        print("\033[97m {}\033[00m" .format('Generated Code : \n{i}'.format(i=code)))
        li = []
        li.append({
            'code' : code,
        })
        return   li
    

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")