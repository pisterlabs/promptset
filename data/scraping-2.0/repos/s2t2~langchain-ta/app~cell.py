
from langchain.docstore.document import Document

from app.text_splitter import parse_cell_type

EMPTY_CODE_CELL = "'code' cell: '[]'"
EMPTY_TEXT_CELL = "'markdown' cell: '[]'"

class Cell(Document):
    # https://github.com/langchain-ai/langchain/blob/451c5d1d8c857e61991a586a5ac94190947e2d80/libs/core/langchain_core/documents/base.py#L9

    def __init__(self, page_content:str, metadata=None):
        metadata = metadata or {}
        super().__init__(page_content=str(page_content), metadata=metadata, type="Document")

        self.metadata["cell_type"] = parse_cell_type(self.page_content)
        self.metadata["is_empty"] = self.is_empty

    @property
    def cell_type(self):
        return self.metadata["cell_type"]

    @property
    def is_code(self):
        return bool(self.cell_type == "CODE")

    @property
    def is_text(self):
        return bool(self.cell_type == "TEXT")

    @property
    def is_empty(self):
        return bool(self.page_content.strip() in [EMPTY_CODE_CELL, EMPTY_TEXT_CELL])
