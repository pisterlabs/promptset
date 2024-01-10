from tools.base_tool import BaseTool
from typing import Dict, List
import inspect
import pkgutil
import importlib
import traceback
from openai import OpenAI
from colorama import Fore
import importlib
import logging
import json
import re


class ToolManager:
    def __init__(self, package='tools', model='gpt-4-1106-preview'):
        tools_package = importlib.import_module(package)
        self.logger = logging.getLogger(__name__)
        self.logger.info('ToolManager initialized') # Added logging
        self.tools: Dict[str, BaseTool] = {}
        self.client = None
        self.model = model
        self.load_tools(tools_package)

    # Split the discovery into two phases
    def load_tools(self, tools_package):
        atomic_tools, complex_tools = self.discover_tools(tools_package)

        # Load atomic tools first
        for name, tool_cls in atomic_tools.items():
            self.tools[name] = tool_cls(self)
        
        # Load complex tools that have (atomic) dependencies
        for name, tool_cls in complex_tools.items():
            if self.check_dependencies(tool_cls.dependencies):
                self.tools[name] = tool_cls(self)
            else:
                missing = [dep for dep in tool_cls.dependencies if dep not in self.tools]
                error_message = f"Failed to load tool '{name}': Missing dependencies {missing}"
                # You can print, log or handle this error message as appropriate for your application
                self.logger.error(error_message)
        
        self.logger.info(f"Loaded tools: {self.tools}")
        
    # Check if specified dependencies are satisfied
    def check_dependencies(self, dependencies: List[str]) -> bool:
        return all(dep in self.tools for dep in dependencies)

    @staticmethod
    def parse_docstring(docstring):
        """
        Manual parsing of the docstring to extract short description and parameter descriptions.
        """
        lines = iter(docstring.strip().split('\n'))
        short_description = next(lines, '').strip()
        param_descriptions = {}
        
        # A regular expression pattern to match parameter declarations in the docstring
        param_pattern = re.compile(r'^(\w+)\s+\((\w+)\)\s*:\s*(.*)')

        current_param = None
        current_description = []

        for line in lines:
            line_stripped = line.strip()
            param_match = param_pattern.match(line_stripped)

            if line_stripped.startswith('Args:'):
                continue
            elif param_match:
                # Save previous argument's description
                if current_param:
                    param_descriptions[current_param] = ' '.join(current_description).strip()
                # Start a new parameter description
                current_param = param_match.group(1)
                current_description = [param_match.group(3).strip()]
            elif line_stripped and current_param:
                # Continue current parameter description
                current_description.append(line_stripped)
            elif not line_stripped and current_param:
                # Allow one blank line in the description but terminate on the next
                current_description.append('')
        
        # Capture description of the last argument
        if current_param:
            param_descriptions[current_param] = ' '.join(current_description).strip()

        return short_description, param_descriptions

    @staticmethod
    def generate_json_for_tool(tool):
        """Generate JSON representation for the tool by introspecting the execute method."""

        execute_function = tool.execute
        sig = inspect.signature(execute_function)
        docstring = inspect.getdoc(execute_function)
        short_description, param_descriptions = ToolManager.parse_docstring(docstring)
        parameters = sig.parameters

        json_definition = {
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": short_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        for param_name, param in parameters.items():
            if param_name == 'self':  # Skip 'self'
                continue
            param_description = param_descriptions.get(param_name, "")
            json_definition["function"]["parameters"]["properties"][param_name] = {
                "type": "string",  # Assume string type for simplicity, adjust if needed
                "description": param_description
            }
            if param.default is inspect.Parameter.empty:
                json_definition["function"]["parameters"]["required"].append(param_name)

        return json_definition

    def get_tools_json(self):
        return [
            self.generate_json_for_tool(tool.__class__) for tool in self.tools.values()
        ]

    # Separate discovery into atomic and complex tools
    def discover_tools(self, tools_package):
        atomic_tools = {}
        complex_tools = {}
        for finder, name, ispkg in pkgutil.iter_modules(
            tools_package.__path__,
            tools_package.__name__ + "."):
            if not ispkg:
                module = importlib.import_module(name)
                for member_name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseTool) and member_name != 'BaseTool':
                        if not getattr(obj, 'dependencies', []):  # Atomic tool with no dependencies
                            atomic_tools[member_name] = obj
                        else:
                            complex_tools[member_name] = obj

        return atomic_tools, complex_tools
        
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        try:
            tool_obj = self.tools[tool_name]
            return tool_obj.execute(**kwargs)
        except Exception as e:
            return f"Error executing {tool_name}: {traceback.format_exc()} {e}"


    def get_response(self, messages: List[Dict[str, str]]) -> str:
        if self.client is None:
            self.client = OpenAI()
        
        self.logger.debug(f"Creating ChatCompletion for messages: {messages}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.get_tools_json(),
            tool_choice='auto'
        )

        completion_messages = response.choices[0].message
        messages.append(completion_messages)
        tool_call_responses = []
        # Process each tool call suggested by the GPT response

        if completion_messages.tool_calls:

            for tool_call in completion_messages.tool_calls:
                tool_name = tool_call.function.name
                function_args = tool_call.function.arguments

                # Execute the tool
                if input(f"{Fore.MAGENTA}{tool_name}({function_args}) ? (y/n) ").lower() == 'n':
                    result = f"User rejected tool usage!"
                else:
                    print(f"{Fore.MAGENTA}=>")
                    try:
                        kwargs = json.loads(function_args)
                    except Exception as e:
                        print(f'{Fore.RED}Error parsing JSON {function_args=}: {e}')
                        kwargs = {}

                    result = self.execute_tool(tool_name, **kwargs)
                    print(f"{Fore.MAGENTA}{result}")

                if result is None:
                    result = "None"

                # Prepare response
                tool_response = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result) if isinstance(result, dict) else result
                }
                tool_call_responses.append(tool_response)
            
            messages.extend(tool_call_responses)

            second_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages # disallow repeated function calls?
            )
            return second_response.choices[0].message
        
        else:
            return completion_messages
