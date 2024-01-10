import requests
import re
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MarkdownifyModel(BaseModel):
    text: str = Field(..., description="Text to be formatted")
    bold: bool = Field(False, description="Whether to bold the text")
    italic: bool = Field(False, description="Whether to italicize the text")
    code: bool = Field(False, description="Wether to format the text as code")
    
class MarkdownifyTool(BaseTool):
    name = "markdownify"
    description = "A tool to format text in markdown. It can make text bold, italic, bold-italic or format it as code."
    args_schema: Type[MarkdownifyModel] = MarkdownifyModel

    def _run(self, text: str, bold: bool = False, italic: bool = False, code: bool = False):
        if not text:
            return text
        
        if bold:
            text = "**" + text + "**"
        if italic:
            text = "*" + text + "*"
        return text
        
    
    def _arun(self, id: str):
        raise NotImplementedError("MarkdownifyTool is not implemented using async")
