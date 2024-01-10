from langchain.tools import BaseTool
from typing import Optional, List, Any 
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from utils.llm_utility import llm
from utils.get_args import fill_signature

class GetSimilarWorkItems(BaseTool):
    name = "get_similar_work_items"
    description = '''
    
    USAGE :
        - Use this tool when you want to get work_items similar to the current work_item.
        - This tool returns a list of similar work_items for the given work_id. 
    '''
    bag_of_words = set(["similar","similar items", "similar work_items", "similar work"])
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        print('\ninside get_similar_work_items tool...') 
        signature = {
                        'work_id': str,
                    }
        arg_description = {
            'work_id': 'The ID of the work item for which you want to find similar items',
        }
        li = []
        for key, value in signature.items():
            arg_dtype = {
                'argument_name': key,
                'argument_value': value,
            }
            arg_descr = {
                'argument_name': key,
                'argument_value': arg_description[key],
            }
            x = fill_signature(query = query, arg_name = key , arg_dtype = arg_dtype , arg_descr = arg_descr, tool_name = self.name)
            if x is not None:
                li.append({
                    'argument_name': key,
                    'argument_value': x,
                })
       
        print('Extracted arguments are : ',li)
        return   li
    

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
