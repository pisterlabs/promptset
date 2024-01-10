import logging
from pathlib import Path

from repo_gpt.agents.base_agent import BaseAgent
from repo_gpt.file_handler.generic_code_file_handler import PythonFileHandler
from repo_gpt.openai_service import OpenAIService
from repo_gpt.search_service import SearchService

logger = logging.getLogger(__name__)


class CodeWritingAgent(BaseAgent):
    system_prompt = """You are an expert software engineer writing code in a repository. The user gives you a plan detailing how the code needs to be updated. You implement the code changes using functions. Ask clarifying questions.
     **DO NOT** respond to the user directly. Use the functions instead.
    """

    def __init__(
        self,
        user_task,
        root_path,
        embedding_path,
        system_prompt=system_prompt,
        threshold=10,
        debug=False,
        openai_key=None,
    ):
        self.system_prompt = system_prompt if system_prompt else self.system_prompt
        super().__init__(
            user_task, "completed_all_code_updates", system_prompt, threshold, debug
        )  # Call ParentAgent constructor
        self.root_path = root_path
        self.embedding_path = embedding_path
        self.openai_service = (
            OpenAIService() if not openai_key else OpenAIService(openai_key)
        )
        self.search_service = SearchService(self.openai_service, self.embedding_path)
        self.codefilehandler = (
            PythonFileHandler()
        )  # TODO: update to handle more than python files (all except sql)

        self.functions = self._initialize_functions()

    def _initialize_functions(self):
        return [
            {
                "name": "create_file",
                "description": "Create a new file with the provided content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the new file to be created.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write in the new file.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "append_to_file",
                "description": "Append content to an existing file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to be updated.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append to the file.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "completed_all_code_updates",
                "description": "Call this function when all the code updates are completed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_changes": {
                            "type": "string",
                            "description": "Enumeration of all the changes that were made to the code.",
                        }
                    },
                    "required": ["code_changes"],
                },
            },
        ]

    def completed_all_code_updates(self, code_changes):
        self.append_function_result_message("completed_all_code_updates", code_changes)
        return code_changes

    def create_file(self, file_path, content):
        """
        Create a new file with the provided content.

        Args:
        - file_path (str): Path to the new file to be created.
        - content (str): Content to write in the new file.

        Returns:
        - str: Success or error message.
        """
        full_path = self.root_path / Path(file_path)

        # Check if file already exists
        if full_path.exists():
            return (
                f"File {file_path} already exists. To update it, use append_to_file()."
            )

        with open(full_path, "w") as f:
            f.write(content)

        return f"File {file_path} has been created successfully."

    def append_to_file(self, file_path, content):
        """
        Append content to an existing file.

        Args:
        - file_path (str): Path to the file to be updated.
        - content (str): Content to append in the file.

        Returns:
        - str: Success or error message.
        """
        full_path = self.root_path / Path(file_path)

        # Check if file exists
        if not full_path.exists():
            return f"File {file_path} does not exist. To create it, use create_file()."

        with open(full_path, "a") as f:
            f.write(content)

        return f"Content has been appended to {file_path} successfully."
