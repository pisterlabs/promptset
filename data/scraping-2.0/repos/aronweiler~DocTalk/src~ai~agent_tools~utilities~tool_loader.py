import json

from langchain.agents import load_tools as load_special_tools

import ai.agent_tools.utilities.tool_creator as tool_creator
from ai.agent_tools.utilities.tool_header import ToolHeader


def load_tools(config, memory, override_llm):
    tools = []

    for tool_json in config["tools"]:
        if "special" in tool_json:
            # special case
            special_tools = load_special_tools([tool_json["special"]])
            for tool in special_tools:
                tools.append(tool)
        else:
            # Load the JSON into the specified type, getting the tool back        
            tool = tool_creator.create_tool(
                tool_json, memory, override_llm
            )

            if tool is not None:
                tools.append(tool)

    return tools


def load_tools_from_file(tool_config_path, memory, override_llm):
    with open(tool_config_path) as config_file:
        config = json.load(config_file)

    return load_tools(config, memory, override_llm)
