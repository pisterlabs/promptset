from langchain.tools.base import BaseTool
import os
from pydantic import Field

class TextFileWriterTool(BaseTool):
    """Tool that writes input to a text file."""

    name = "TextFileWriterTool"
    description = (
        "A tool that writes input to a text file. "
        "Input should be a summarized text of no more than 1000 characters."
    )
    
    # Declare file_path as a class field
    file_path: str = Field(default="")

    def __init__(self) -> None:
        super().__init__()  # Call the parent constructor
        self.file_path = os.getenv('MY_FILE_PATH')
        if self.file_path is None:
            raise ValueError("Environment variable 'MY_FILE_PATH' is not set.")
        return

    def _run(self, tool_input: str) -> str:
        """Use the TextFileWriterTool."""
        # Check if input is less than or equal to 1000 characters
        if len(tool_input) > 1000:
            return "Error: Input text is longer than 500 characters."

        # Append the input to the text file
        with open(self.file_path, 'a') as f:
            f.write(tool_input + '\n')

        return "Text successfully appended to the file."

    async def _arun(self, tool_input: str) -> str:
        """Use the TextFileWriterTool asynchronously."""
        return self._run(tool_input)
