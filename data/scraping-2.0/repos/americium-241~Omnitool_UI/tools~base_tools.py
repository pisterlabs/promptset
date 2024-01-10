import inspect
from storage.logger_config import logger
from langchain.tools import BaseTool
import langchain.tools as ltools
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.agents import load_tools
import streamlit as st 
import os 

Local_dir = os.path.dirname(os.path.realpath(__file__))

class Ui_Tool(BaseTool):
    name = 'Base_tool'
    link = r'https://github.com/americium-241/Omnitool_UI/tree/master'
    icon = '🔧 '
    description = 'Description'

    def _run(self,a):
        """This function should be overwrite when creating a tool and a docstring have to be given"""
        logger.debug('You are in the Base tool execution and I inputed :',a)
        return 'Success'

    def _ui(self):
        # Overwrite this function to add options to the tool, use streamlit_extra mui components
        pass
    

# Your existing function to check if a class has required attributes
def has_required_attributes(cls):
    """Check that class possesses a name and description attribute"""
    required_attributes = ['name', 'description']
    try:
        instance = cls(**{attr: "default_value" for attr in required_attributes})
        return True
    except Exception as e:
        #print(f"Failed to instantiate {cls.__name__} due to: {e}")
        return False

def make_pre_structured_tools(): 
   
    """Monitoring langchains.tools and keeping only tools without any mandatory arguments for initialisation"""       
    module = ltools
    tool_class_names = [member for name, member in inspect.getmembers(module) if isinstance(member, list)][0]
    # Retrieve each class using its name and check if it has the required attributes
    classes = [getattr(module, class_name) for class_name in tool_class_names]
    
    p_tools = [cl() for cl in classes if has_required_attributes(cl)]
    pre_tools= []
    toolkit_file = FileManagementToolkit(root_dir=Local_dir+"\\..\\workspace") 
    pre_tools.extend(toolkit_file.get_tools())
    tools_name=[t.name for t in pre_tools]
    for t in p_tools : 
        if t not in pre_tools and t.name != 'DuckDuckGo Results JSON' and t.name not in tools_name:
            pre_tools.append(t)
    requests_tools = load_tools(["requests_all"])
  
 
    pre_tools.extend(requests_tools)
  
    return pre_tools



Pre_Structured_Tool=make_pre_structured_tools()

