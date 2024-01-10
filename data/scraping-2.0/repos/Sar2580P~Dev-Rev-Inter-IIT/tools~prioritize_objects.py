from langchain.tools import BaseTool
from typing import Optional, List, Any 
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from utils.llm_utility import llm
from utils.get_args import fill_signature

class Prioritize(BaseTool):
    name = "prioritize_objects"
    description = '''
    - Use this tool when asked to prioritize the objects. 
    '''
                
    bag_of_words = set(["prioritize", "priority", "prioritize objects", "prioritization"])
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        print('\ninside Prioritize_objects tool...') 
        signature = {
                        'objects': List[str],
                    }

        arg_description = {
            'objects': 'the list of objects to be prioritized',
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
            query = query.strip('\n').strip()
            ans = query    ## $$PREV[*] is a special keyword that means "use the previous value of this argument"

            if  len(query) != 9:
                ans = fill_signature(query = query, arg_name = key , arg_dtype = arg_dtype , arg_descr = arg_descr, tool_name = self.name)
                
            if ans.strip('\n').strip() != 'NONE':
                li.append({
                    'argument_name': key,
                    'argument_value': ans,
                })
       
        print('Extracted arguments are : ',li)
        return   li
    
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")