
tool_name = input("Tool Name")
tool_desc = input("Tool description")

arg_names = []
arg_descs = []
arg_types = []

while True:
    arg_name = input("Argument name (type 'exit' if done): ").strip()
    if arg_name.lower() == "exit":
        break
    arg_desc = input("Argument description: ").strip()
    arg_type = input("Argument type: ").strip()

    arg_names.append(arg_name)
    arg_descs.append(arg_desc)
    arg_types.append(arg_type)

tool_arg_dict = dict(zip(arg_names, arg_types))
tool_arg_descript_dict = dict(zip(arg_names, arg_descs))

NEW_TOOL_CLASS = """
from langchain.tools import BaseTool
from typing import Optional, Type, List, Any
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from tools.argument_mapping.get_args import fill_signature
from backend_llm.utils import llm


class {tool_name}(BaseTool):
    name = {tool_name}
    description = '''This tool is useful for {tool_description}:  
        
        The arguments:
    ''' + ", ".join({tool_arguments})

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        signature = {tool_arg_dict}
        
        arg_description = {tool_arg_descript_dict}
        column_args = fill_signature(query,function_signatures= signature , arg_description=arg_description,tool_name=self.name)
        li = []
        for key, value in column_args.items():
          x = {{
              'argument_name': 'key',
              'argument_value': 'value',
          }}
          li.append(x)
        return   li

    
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        "Use the tool asynchronously."
        raise NotImplementedError("custom_search does not support async")
""".format(tool_name=tool_name, tool_description=tool_desc, tool_arguments=arg_names,
           tool_arg_dict=tool_arg_dict, tool_arg_descript_dict=tool_arg_descript_dict)

with open('tools/{tool_name}.py'.format(tool_name=tool_name),"w") as f:
    f.write(NEW_TOOL_CLASS)
    f.close()






