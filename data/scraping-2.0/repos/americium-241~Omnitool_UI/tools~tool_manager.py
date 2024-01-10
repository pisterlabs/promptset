import os 
from langchain.tools import StructuredTool

from .base_tools import Pre_Structured_Tool
from .utils import get_class_func_from_module, monitorFolder


class ToolManager:

    def __init__(self):
        self.structured_tools = make_tools_list()
        self.tools_ui={}
        self.tools_description=self.make_tools_description()

    def make_tools_description(self): 
        tools_description = {}
        for t in  self.structured_tools : 
            tools_description.update({t.name : t.description})
        return tools_description
          
    def get_tools(self):
        return self.structured_tools
    
    def get_tool_names(self):
        return [tool.name for tool in self.structured_tools]

    def get_selected_tools(self, selected_tool_names):
        return [tool for tool in self.structured_tools if tool.name in selected_tool_names]


def make_tools_list():
 
    #Define the path of the monitored folder for tool auto listing
    monitored_files=monitorFolder(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools_list'))
    pre_tool = Pre_Structured_Tool # get a list of structured tools 
    base_tool_list=[]

    for mod in monitored_files:
        #Get the lists of tools to construct fro tool_list.py
        listClassTool,listFunctionTool=get_class_func_from_module(mod)
        
        # ListClassTool and ListFunctionTool are created on the fly from monitored files
        for b_tool in listClassTool:
            base_tool_list.append(b_tool[1]())

        for func in listFunctionTool :
            functool=StructuredTool.from_function(func[1])
            base_tool_list.append(functool)

    for pre_tool in Pre_Structured_Tool :
        base_tool_list.append(pre_tool)
  
    return base_tool_list
