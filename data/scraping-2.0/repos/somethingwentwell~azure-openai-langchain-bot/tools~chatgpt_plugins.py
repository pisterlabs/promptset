from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool
from typing import List

def import_plugins_from_file(filepath: str, tools: List[AIPluginTool]) -> List[AIPluginTool]:
    with open(filepath, "r") as f:
        for line in f:
            plugin_url = line.strip()
            tool = AIPluginTool.from_plugin_url(plugin_url)
            tools.append(tool)
    return tools

chatgpt_tools = import_plugins_from_file("./tools/chatgptplugins.txt", [ ])

def chatgpt_plugins():
    tools = load_tools(["requests_all"])
    tools.extend(chatgpt_tools)
    return tools