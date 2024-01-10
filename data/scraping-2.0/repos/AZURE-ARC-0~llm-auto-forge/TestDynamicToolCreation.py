from langchain import WikipediaAPIWrapper
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from prompts import TOOL_MAKER_PROMPT
from Agents import DialogueAgentWithTools

import util
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools.file_management import WriteFileTool, ReadFileTool
from tools.ToolRegistrationTool import tool_registration_tool
from tools.ToolQueryTool import tool_query_tool

util.load_secrets()

# Define system prompts for our agent
system_prompt_scribe = TOOL_MAKER_PROMPT

tools = [ReadFileTool(),
         WriteFileTool(),
         WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
         DuckDuckGoSearchRun(),
         tool_registration_tool,
         tool_query_tool]

# Initialize our agents with their respective roles and system prompts
tool_making_agent = DialogueAgentWithTools(name="ToolMaker",
                                           system_message=system_prompt_scribe,
                                           model=ChatOpenAI(
                                               model_name='gpt-4',
                                               streaming=True,
                                               temperature=0.9,
                                               callbacks=[StreamingStdOutCallbackHandler()]),
                                           tools=tools)

tool_making_agent.receive("HumanUser", "Write the first sentence of the gettysburg address to a file (create a tool to do this).")

tool_making_agent.send()

print("Done")
