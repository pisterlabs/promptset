# /Tools/ReadFile.py

import os
from instructor import OpenAISchema
from pydantic import Field
from Utilities.Config import WORKING_DIRECTORY
from Utilities.Log import Log, type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agency.Agency import Agency

class ReadFile(OpenAISchema):
    """
    Read the contents of a local file.
    """
    file_name: str = Field(
        ..., description="The name of the file including the extension"
    )
    directory: str = Field(
        default=WORKING_DIRECTORY,
        description="The path to the directory where to file is stored. Path can be absolute or relative."
    )

    def run(self, agency: 'Agency'):
        
        # If file doesnt exist, return message
        if not os.path.exists(self.directory + self.file_name):
            result = f"File {self.directory + self.file_name} does not exist."
            Log(type.ERROR, result)
            return result
        
        # todos: 
        # 1. upload file to openai
        # 2. store file id somewhere in agency
        # 3. attach file id to a new message & drop in queue
        # 4. create new run with file id
        # 5. return tool output?
        # 6. wait for in progress and cancel run?
        # 7. start the run with the file id
        # 8. test and review
        
        Log(type.ACTION, f"Viewing content of file: {self.directory + self.file_name}")
        
        with open(self.directory + self.file_name, "r") as f:
            file_content = f.read()

        return f"File contents:\n{file_content}"