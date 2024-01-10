# /Tools/MoveFile.py

import os
from instructor import OpenAISchema
from pydantic import Field
from Utilities.Config import WORKING_DIRECTORY
from Utilities.Log import Log, type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agency.Agency import Agency


class MoveFile(OpenAISchema):
    """
    Move a file from one directory to another.
    This tool can also be used to rename a file when the destination is the same directory but the supplied new file name is different.
    """

    file_name: str = Field(
        ..., description="Required: The name of the file (including the extension) to be moved"
    )
    new_name: str = Field(
        default="",
        description="Optional: The new name of the file (including the extension) to be named at the new destination. If left blank, the file will be moved with the same name.",
    )
    directory: str = Field(
        default=WORKING_DIRECTORY,
        description="Optional: The path to the directory where the file is currently stored. Path can be absolute or relative.",
    )
    destination: str = Field(
        default="",
        description="Optional: The path to the directory where to file is be moved to. Path can be absolute or relative."
    )

    def run(self, agency: 'Agency'):
        
        # If file doesnt exist, return message
        if not os.path.exists(self.directory + self.file_name):
            return f"File {self.directory + self.file_name} does not exist."
        
        if self.new_name == "":
            self.new_name = self.file_name
            
        if self.destination == "":
            self.destination = self.directory

        file_destination_path = os.path.join(self.destination, self.new_name)
        os.rename(self.directory + self.file_name, file_destination_path)

        # if destination is the same but file name is different, log rename
        if self.directory == self.destination and self.new_name != self.file_name:
            result = f"File renamed to: {file_destination_path}"
        else:
            result = f"File moved to: {file_destination_path}"
            
        Log(type.RESULT, result)
        return result
