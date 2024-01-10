from abc import ABC
from pathlib import Path
from typing import List

from langchain import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.tools.file_management import (
    ReadFileTool,
    ListDirectoryTool,
)
from langchain.tools.human.tool import HumanInputRun

from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, WORKSPACE_DIR
from command_gpt.utils.custom_stream import CustomStreamCallback
from command_gpt.tooling.tools import SearchAndWriteTool, WriteFileToolNewlines

WORKSPACE_PATH = Path(WORKSPACE_DIR)


class BaseToolkit(ABC):
    """
    Base Toolkit with tools initialized for the project, stored in a dict.
    - Use get_toolkit() to get the List[BaseTool] for this toolkit.
    - Available Tools: ["search", "write_file", "read_file", "list_directory", "finish", "human_input"]
    """

    def __init__(self):
        super().__init__()

        # region Search/Web
        # - Custom Google Search API Wrapper used by SearchAndWriteTool to run a search query and automatically write the results to a file (saving resources)
        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID,
        )
        search_and_write = SearchAndWriteTool(search)
        search_tool = Tool(
            name="search",
            func=search_and_write.run,
            description="Gather search results from query (they will automatically be written to a file called 'results_{query}').",
            callbacks=[CustomStreamCallback()]
        )

        # endregion
        # region File Management
        # - Tools for reading, writing, and listing files in the workspace directory
        # - Note: WriteFileToolNewlines is a simple extension of WriteFileTool that re-writes new line characters properly for file writing
        write_file_tool = WriteFileToolNewlines(
            root_dir=WORKSPACE_DIR,
            callbacks=[CustomStreamCallback()]
        )
        read_file_tool = ReadFileTool(
            root_dir=WORKSPACE_DIR,
            callbacks=[CustomStreamCallback()]
        )
        list_directory_tool = ListDirectoryTool(
            root_dir=WORKSPACE_DIR,
            description="List files to read from or append to.",
            callbacks=[CustomStreamCallback()]
        )

        # endregion
        # region Other
        finish_tool = Tool(
            name="finish",
            func=lambda: None,
            description="End the program.",
            callbacks=[CustomStreamCallback()]
        )
        human_input_tool = Tool(
            name="human_input",
            func=HumanInputRun,
            description="Get input from a human if you find yourself overly confused.",
            callbacks=[CustomStreamCallback()]
        )

        # endregion
        # region TOOLS DICTIONARY
        # - Adds tools to dictionary accessible by name

        self.tools = {
            'search': search_tool,
            'write_file': write_file_tool,
            'read_file': read_file_tool,
            'list_directory': list_directory_tool,
            'finish': finish_tool,
            'human_input': human_input_tool
        }

        # endregion

    def get_toolkit(self) -> List[BaseTool]:
        """
        Return the list of tools for this toolkit.
        """
        return list(self.tools.values())


class MemoryOnlyToolkit(BaseToolkit):
    """
    Basic Toolkit with only file management tools (no search/web)
    - Available tools: ["write_file", "read_file", "list_directory", "finish", "human_input"]
    """

    def get_toolkit(self) -> List[BaseTool]:
        """
        Return a modified list of tools for this toolkit, excluding search/web tools.
        """
        return [tool for tool in super().get_toolkit() if tool.name not in ['search']]
