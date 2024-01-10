from langchain.tools import BaseTool
from typing import Optional, Type, List, Any, Tuple
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from utils.get_args import *
from utils.llm_utility import llm
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


class WorkList(BaseTool):
    name = "works_list"
    description = '''
    - This tool can search and return the relevant work-items on the basis of various search filters.
    - Below are the arguments present which can be used to filter the work-items .
    - Whenever the query contains below arguments as keywords, give this tool a try.
    
    Following are the possible arguments with their description that the tool can take -->
        - 'applies_to_part': for accessing work items applicable to a particular part of the project.
        - 'created_by': for accessing the work items created by a particular person.
        - 'issue.priority': For accessing work_items with issues of a particular priority. Can be either of types --> "p0" , "p1" , "p2".
        - 'issue.rev_orgs': For accessing the work-items with issues of provided rev_orgs.
        - 'limit' : Limitig the maximum no. of work itmes to return back, DEFAULT : "all" 
        - 'owned_by': Accessing the work items owned by a particular user id
        - 'stage.name': Accessing work items belonging to a particular stage
        - 'ticket.needs_response': Accessing work_items with tickets that needs response or not, must be either True or False,
        - 'ticket.rev_org': Accessing work_items with ticket belonging to a particular rev_org
        - 'ticket.severity': Accessing work items on the basis of ticket severity. MUST BE ONE OF --> 'blocker' , 'high' , 'medium' , 'low',
        - 'ticket.source_channel': Accessing the work-items with tickets belonging to the provided source channel
        - 'type': Accessing work-items on the basis of type, MUST BE one of --> 'issues', 'ticket' , 'task'
    '''

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        print('\ninside work_list tool...')
        

        filtered_signature , filtered_arg_description = self._filtered_arguments(query)
        li = []
        for key, value in filtered_signature.items():
            arg_dtype = {
                'argument_name': key,
                'argument_value': value,
            }
            arg_descr = {
                'argument_name': key,
                'argument_value': filtered_arg_description[key],
            }
            x = fill_signature(query = query, arg_name = key , arg_dtype = arg_dtype , arg_descr = arg_descr, tool_name = self.name)
            if 'NONE' in x:
                continue
            # if filtered_signature[key] == List[str]:
            #     print('ooooooooooooooooooooo')
            #     if x[0] != '[':
            #         x = '[' + x + ']'
            # if x.strip('\n').strip() != 'NONE':
            li.append({
                'argument_name': key,
                'argument_value': x,
            })
        
        print('Extracted arguments are : ',li)
        return   li

#____________________________________________________________________________________________________________________________________
    def _filtered_arguments(self, query: str) -> Tuple[dict, dict]:
        """Returns the filtered arguments and their descriptions."""

        signature = {
                    'applies_to_part' : List[str] ,
                    'created_by': List[str] ,
                    'issue.rev_orgs': List[str] ,
                    'owned_by': List[str] ,
                    'ticket.needs_response': bool ,
                    'ticket.rev_org': List[str] ,
                    'ticket.source_channel': str ,
                   }
        
        arg_description = {
         'applies_to_part': 'for accessing work items applicable to a particular part of the project.',
         'created_by': 'for accessing the work items created by a particular person.',
         'issue.priority':' For accessing work_items with issues of a particular priority. Can be either of types --> "p0" , "p1" , "p2".' ,
         'issue.rev_orgs': 'For accessing the work-items with issues of provided rev_orgs.',
         'limit' : 'Limiting the maximum no. of work items to return back, DEFAULT : "all" ',
         'owned_by': 'Accessing the work items owned by a particular user id',
         'stage.name':' Accessing work items belonging to a particular stage',
         'ticket.needs_response': 'Accessing work_items with tickets that needs response or not, must be either "True" or "False"',
         'ticket.rev_org':' Accessing work_items with ticket belonging to a particular rev_org',
         'ticket.severity': "Accessing work items on the basis of ticket severity. MUST BE ONE OF --> 'blocker' , 'high' , 'medium' , 'low'," ,
         'ticket.source_channel':' Accessing the work-items with tickets belonging to the provided source channel',
         'type': "Accessing work-items on the basis of type, MUST BE one of --> 'issues', 'ticket' , 'task'"
        }

        filtered_signature = {}
        filtered_arg_description = {}

        query = query.lower().strip()
        arguments = arg_description.keys()
        if 'p0'in query or 'p1'in query or 'p2'in query:
            filtered_signature['issue.priority'] = str
            filtered_arg_description['issue.priority'] = arg_description['issue.priority']

        if 'all' in query or 'limit..\b' in query:
            filtered_signature['limit'] = int
            filtered_arg_description['limit'] = arg_description['limit']

        if 'issues' in query or  'ticket' in query or  'task' in query:
            filtered_signature['type'] = List[str]
            filtered_arg_description['type'] = arg_description['type']

        if 'blocker' in query or 'high' in query or 'medium' in query or 'low' in query:
            filtered_signature['ticket.severity'] = List[str]
            filtered_arg_description['ticket.severity'] = arg_description['ticket.severity']

        if 'stage' in query:
            filtered_signature['stage.name'] = List[str]
            filtered_arg_description['stage.name'] = arg_description['stage.name']

        if 'channel' in query.lower():
            filtered_signature['ticket.source_channel'] = str
            filtered_arg_description['ticket.source_channel'] = arg_description['ticket.source_channel']


        x = set(filtered_signature.keys())
        x.add('issue.priority')
        x.add('limit')
        x.add('ticket.source_channel')
        x.add('type')
        x.add('ticket.severity')
        x.add('stage.name')

        remaining_arguments = set(arguments) - x

        remaining_arguments =  filter_arguments(query, remaining_arguments, arg_description)
        for arg in remaining_arguments:
            filtered_signature[arg] = signature[arg]
            filtered_arg_description[arg] = arg_description[arg]

        print(filtered_arg_description)
        return filtered_signature, filtered_arg_description

#____________________________________________________________________________________________________________________________________
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
