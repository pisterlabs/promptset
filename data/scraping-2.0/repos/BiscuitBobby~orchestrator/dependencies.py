# common dependencies
import os
import dotenv
from langchain.tools import BaseTool  # Import the correct BaseTool
from langchain.llms import GooglePalm
from langchain.tools.render import render_text_description
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain import hub
# https://github.com/hwchase17/chroma-langchain/blob/master/qa.ipynb <- example of using chroma
