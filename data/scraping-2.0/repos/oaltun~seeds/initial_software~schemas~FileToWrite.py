from typing import Optional
import pydantic
from pathlib import Path
from pydantic import validator

from instructor import OpenAISchema

class FileToWrite(OpenAISchema):
    "Content and path for a file to write. If file already exists, it is overwritten."
    path: Optional[str]=""
    content: Optional[str] = ""
  