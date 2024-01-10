from IPython.core.magic import magics_class, register_line_cell_magic, Magics

import ast
import argparse
import astor
import folium
import pandas as pd
import re
import os

from copy import deepcopy

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain.utilities import GoogleSerperAPIWrapper


from typing import Any, Optional, Dict, List, Union

from constants import WHITELISTED_LIBRARIES, WHITELISTED_BUILTINS


class ChatAgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.descriptions = []
        self.agent_action = None

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print("on action")
        index = action.log.find("Action:")
        self.agent_action = action
        if index != -1:
            self.descriptions.append(action.log[:index].strip())


@magics_class
class ChatAgentMagics(Magics):
    def __init__(self):
        # super(ChatAgentMagics, self).__init__(shell)  uncomment this when making this a proper jupyter extension loaded with %load_ext
        self.__agent_input = {}
        self.__llm = OpenAI(temperature=0)
        tools = (
            load_tools(["google-serper"], llm=self.__llm)
            if "SERPER_API_KEY" in os.environ
            else []
        )
        self.__tools = tools + [
            Tool.from_function(
                func=self.python_execution,
                name="pythonCodeExecution",
                description="Tool used to execute Python code. Input should be python code containing statements to derive answers to questions or solutions to instructions. The input code should store the answer in variable named result unless instructed otherwise. The tool may return feedback from the user on the input code. If the result is a numeric value be sure to assign it to a variable with proper formatting without commas, dollar signs,  percent symbols or any other symbol.",
            ),
            Tool.from_function(
                func=self.plot_folium_map,
                name="mapPlottingTool",
                description="Tool used to plot markers on a map. Input to the tool should be the name of a Pandas dataframe that has the columns name, latitude, and longitude.",
            ),
        ]
        self.__agent = initialize_agent(
            self.__tools,
            self.__llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3,
        )
        self.__callback_handler = ChatAgentCallbackHandler()
        self.__noninteractive = False
        self.__verbose = False
        self.__last_key = None
        self.__return_type_list = [folium.folium.Map]

    def is_df_overwrite(self, node: ast.stmt) -> str:
        """
        Remove df declarations from the code to prevent malicious code execution. A helper method.
        Args:
            node (object): ast.stmt

        Returns (str):

        """

        return (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and re.match(r"df\d{0,2}$", node.targets[0].id)
        )

    def is_unsafe_import(self, node: ast.stmt) -> bool:
        """Remove non-whitelisted imports from the code to prevent malicious code execution

        Args:
            node (object): ast.stmt

        Returns (bool): A flag if unsafe_imports found.

        """

        return isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.name not in WHITELISTED_LIBRARIES for alias in node.names
        )

    def clean_code(self, code: str) -> str:
        """
        A method to clean the code to prevent malicious code execution
        Args:
            code(str): A python code

        Returns (str): Returns a Clean Code String

        """
        tree = ast.parse(code)

        new_body = [
            node
            for node in tree.body
            if not (self.is_unsafe_import(node) or self.is_df_overwrite(node))
        ]

        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree).strip()

    def add_agent_input(
        self,
        input_var: Any,
        name: str,
        description: str,
        rows: int = 5,
        include_df_head: bool = False,
    ):
        if type(input_var) == pd.DataFrame and include_df_head:
            description += f"""
This is the result of `print(df.head({rows}))`
{input_var.head(rows)}
        """
        self.__agent_input[name] = {"value": input_var, "description": description}

    def delete_agent_input(self, key: Union[List[str], str]):
        if type(key) == list:
            keys = key
        else:
            keys = [key]
        for item in keys:
            if item in self.__agent_input:
                del self.__agent_input[item]
            else:
                print(f"Key {item} not found in agent input.")

    def set_agent_input(self, agent_input: dict):
        self.__agent_input = agent_input

    def get_agent_input(self):
        return self.__agent_input

    def get_input_key(self, key_val: str):
        num = len([key for key in self.__agent_input if key_val in key])
        input_key = key_val + (str(num + 1) if num > 0 else "")
        return input_key

    def plot_folium_map(self, df_name: str, limit=20):
        try:
            match = re.match(
                "mapPlottingTool\(([^\W0-9]\w*)\)", df_name
            )  # match incorrect inputs to the tool that use the tool name passing the dataframe as an argument to the tool.
            if match:
                df_name = match.group(1)
            if df_name in self.__agent_input:
                input_df = self.__agent_input[df_name]["value"]
                if type(input_df) == pd.Series:
                    input_df = input_df.to_frame().T
                if "latitude" in input_df.index:
                    input_df = input_df.T
                latitude = (input_df.latitude.max() + input_df.latitude.min()) / 2.0
                longitude = (input_df.longitude.max() + input_df.longitude.min()) / 2.0
                m = folium.Map(location=[latitude, longitude], zoom_start=12)
                if input_df.index.shape[0] > limit:
                    print(
                        "dataframe has rows greater than limit of {limit}: {input_df.shape[0]}"
                    )
                    print("mapping only the first {limit} rows")
                for index in input_df.index[:limit]:
                    folium.Marker(
                        input_df.loc[index][["latitude", "longitude"]],
                        tooltip=input_df.loc[index]["name"],
                    ).add_to(m)
                # result_key = self.get_input_key(f"{df_name}_map") prevents overwritting but adds prompt complexity
                result_key = f"{df_name}_map"
                self.__agent_input[result_key] = {
                    "value": m,
                    "description": f"map for dataframe {df_name}",
                }
                self.__last_key = result_key
                return f"map for {df_name} created"
            else:
                return f"name {df_name} not available in environment"
        except Exception as e:
            return f"tool failed with following error: {e}"

    def python_execution(self, analysis_code: str):
        last_character = self.__callback_handler.agent_action.log.strip()[-1]
        if last_character == '"' and not analysis_code.endswith('"'):
            analysis_code += '"'  # replace missing quotes that langchain strips
        try:
            analysis_code = self.clean_code(analysis_code)
            print()
            if self.__verbose:
                print("input code")
                print(analysis_code)
            user_feedback = ""
            if not self.__noninteractive:
                prompt = f"""
    
    The change agent would like to run the following code:
    
    --------------------------------------------------------
    {analysis_code}
    --------------------------------------------------------
    
    To allow execution type Y or type N to disallow.
    You may give additional feedback for either option by placing a dash after the option followed by the feedback. For example:
    Y - this code answers my original question
    or
    N - this code does not produce the right answer
    
                """
                feedback_retrieved = False
                while not feedback_retrieved:
                    try:
                        user_input = input(prompt)
                        user_input = user_input.strip().split("-")
                        first_input = user_input[0].strip().lower()
                        if first_input not in ("y", "n"):
                            raise ValueError("Must enter Y or N")
                        if len(user_input) > 1:
                            user_feedback = " - ".join(user_input[1:])
                        if first_input == "n":
                            response_end = (
                                "most likely because it doesn't achieve the desired result."
                                if len(user_feedback) == 0
                                else f" and has the following feedback: {user_feedback}"
                            )
                            return f"The user disallowed execution of the code{response_end}"
                        feedback_retrieved = True
                    except ValueError as e:
                        print(e)
                        pass

            input_environment = {
                key: self.__agent_input[key]["value"] for key in self.__agent_input
            }
            environment = {
                **input_environment,
                "__builtins__": {
                    **{
                        builtin: __builtins__[builtin]
                        for builtin in WHITELISTED_BUILTINS
                    },
                },
            }

            exec(analysis_code, environment)

            code_parse = ast.parse(analysis_code, mode="exec")
            key_val = None
            if type(code_parse.body[-1]) == ast.Assign:
                if self.__verbose:
                    print(
                        "The variable `result` was not found in executing environment. Using the assignment on the last code line instead for the result."
                    )
                key_val = code_parse.body[-1].targets[0].id
                result = environment[key_val]
            else:
                return "complete. No assignment operation found in last lines of code."
            # result_key = self.get_input_key(key_val)
            result_key = key_val
            description = f'object of type {type(result)} related to the thought "{self.__callback_handler.descriptions[-1]}"'
            if type(result) == pd.DataFrame:
                description += (
                    f". The dataframe has the columns {result.columns.values}"
                )
            print("saving result to agent input ", result_key)
            self.__agent_input[result_key] = {
                "value": result,
                "description": description,
            }
            response_end = (
                ""
                if len(user_feedback) == 0
                else f" - The user has the following feedback: {user_feedback}"
            )
            self.__last_key = result_key
            return (
                f"Answer has been successfully derived. Key: {result_key}{response_end}"
                if not type(result) == str
                else result + response_end
            )
        except Exception as e:
            return f"execution failed with the error message: {str(e)}"

    def chat_agent(self, line: Optional[str], cell: Optional[str] = None):
        "Magic that works as %%chat_agent"
        options = list(filter(lambda x: len(x) != 0, line.strip().split(" ")))
        parser = argparse.ArgumentParser(description="chat agent options")
        parser.add_argument(
            "--noninteractive",
            "-n",
            action="store_true",
            help="runs the agent in a non interactive mode where the user is not prompted for input",
            default=False,
            required=False,
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="verbose option",
            default=False,
            required=False,
        )
        args = parser.parse_args(options)
        self.__noninteractive = args.noninteractive
        self.__verbose = args.verbose
        available_variables = "\n\n".join(
            [
                key + " - " + self.__agent_input[key]["description"]
                for key in self.__agent_input
            ]
        )
        cell = (
            cell
            + (
                """\nWhen using the pythonCodeExecution tool you may assume that you have access to the following variables when writing the code:

"""
                if len(self.__agent_input) > 0
                else ""
            )
            + available_variables
        )
        cell = cell.strip()
        print("Prompt:")
        print(cell)
        response = self.__agent.run(cell, callbacks=[self.__callback_handler])
        if (
            type(self.__agent_input[self.__last_key]["value"])
            in self.__return_type_list
        ):
            return self.__agent_input[self.__last_key]["value"]
        return response


chat_agent_magic = ChatAgentMagics()

set_inputs = chat_agent_magic.set_agent_input
get_inputs = chat_agent_magic.get_agent_input
add_agent_input = chat_agent_magic.add_agent_input
delete_agent_input = chat_agent_magic.delete_agent_input


def get_result(key: str):
    return get_inputs()[key]["value"]


register_line_cell_magic(chat_agent_magic.chat_agent)

# def load_ipython_extension(ipython):
#     ipython.register_magics(chat_agent_magic)
