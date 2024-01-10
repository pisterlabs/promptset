from langchain.tools import BaseTool
from typing import Optional, Type, List
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from utils.get_args import fill_signature
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.llm_utility import llm


class AddWorkItemsToSprint(BaseTool):
    name = "add_work_items_to_sprint"
    description = '''
    USAGE :
        - Adds or assigns the given work items to the sprint. 
        - Need to fill the following arguments available for tool usage -->
                - "sprint_id" : the id of current sprint
                - "work_ids" : list of work-items to add to sprint

    '''

    bag_of_words = set(["add work items to sprint", "add work items", "add to sprint", " add", "assign", "assigning"])

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) :
        print('\ninside add_work_items_to_sprint tool ...') 
        signature = {'work_ids': List[str],
                    'sprint_id': str ,
                    }
        
        arg_description = {
            'work_ids': 'A list of work item IDs to be added to the sprint',
            'sprint_id': 'The ID of the sprint to which the work items should be added',
            
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
