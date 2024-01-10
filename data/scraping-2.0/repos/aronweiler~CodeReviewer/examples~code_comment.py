from typing import List
from langchain.tools import StructuredTool

class CodeComment:
    """A code comment."""
    
    def __init__(self, comment: str, start: int = None, end: int = None, file_path: str = None):
        self.start = start
        self.end = end
        self.comment = comment
        self.file_path = file_path
