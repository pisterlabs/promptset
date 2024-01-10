from pathlib import Path
from typing import Optional
from schemas.DirFilelistGet import DirFilelistGet
from schemas.FileToWrite import FileToWrite
import pydantic
from pydantic import validator, Field

import instructor
from instructor import OpenAISchema

# instructor.patch() # Enables the response_model 

class Completion(OpenAISchema):
    "AI can ask system for information to understand its environment by file_to_read and dir_filelist_get. AI can also modify/rewrite using file_to_read."
    intent: str = Field(description="Why do you request this? Answer.")
    reminder: str = Field(description="As you have limited number of tokens, and so forget previous conversation, you should output important info to this reminder variable. We will refeed this reminder variable to you on each message.")
    file_to_read: Optional[str] = None
    file_to_write: Optional[FileToWrite] = None
    dir_filelist_get: Optional[DirFilelistGet] = None
