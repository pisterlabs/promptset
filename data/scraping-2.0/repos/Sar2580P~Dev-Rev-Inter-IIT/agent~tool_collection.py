from langchain.tools import BaseTool
from typing import Optional, List, Any
import sys , os
sys.path.append(os.getcwd())
from tools.work_list import WorkList
from tools.summarize_objects import Summarize
from tools.add_work_items_to_sprint import AddWorkItemsToSprint
from tools.create_actionable_tasks_from_text import CreateActionableTasksFromText
from tools.get_similar_work_items import GetSimilarWorkItems
from tools.get_sprint_id import GetSprintId
from tools.prioritize_objects import Prioritize
from tools.search_object_by_name import SearchObjectByName
from tools.who_am_i import WhoAmI
from utils.llm_utility import llm
from tools.logic_tool import LogicalTool
import icecream as ic

task_tools = [
    WhoAmI(),
    SearchObjectByName(),
    GetSprintId(),
    AddWorkItemsToSprint(),
    GetSimilarWorkItems(),
    WorkList() , 
    Summarize() ,
    # LogicalTool(),
    CreateActionableTasksFromText(),
    Prioritize(), 
]

def get_relevant_tools(query: str ) -> List[BaseTool]:
    """Returns the list of relevant tools for the query."""
    relevant_tools = []
    for tool in task_tools:
        if not hasattr(tool, "bag_of_words"):
            relevant_tools.append(tool)
            continue
        # if tool.name == 'search_object_by_name':
        #     relevant_tools.append(tool)
        #     continue
        
        tool_bag_of_words = tool.bag_of_words
        for word in tool_bag_of_words:
            if word in query.lower().strip():
                relevant_tools.append(tool)
                break

    return relevant_tools



# x = get_relevant_tools("Summarize all tickets needing a response in the 'support' rev organization.")

# print(x)
