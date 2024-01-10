from langchain.tools import BaseTool
from typing import Optional, List, Any 
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from utils.llm_utility import llm
from utils.get_args import fill_signature

class GetSprintId(BaseTool):
    name = "get_sprint_id"
    description = '''
    USAGE :
    - This tool is used when we want to know the id of current sprint.
    - Think of using it when user query contains keywords like "sprint"
            '''
    bag_of_words = set(["current sprint", "current sprint id", "sprint", "sprint id"])
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        print('\ninside get_sprint_id tool...') 
        # signature = {}
                        
        # arg_description = {}
        # column_args = fill_signature(query,function_signatures= signature ,arg_description=arg_description, tool_name = self.name)
        li = []
        # for key, value in column_args.items():
        #     x = {
        #         'argument_name': key,
        #         'argument_value': value,
        #     }
        #     li.append(x)
        return li
    

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
