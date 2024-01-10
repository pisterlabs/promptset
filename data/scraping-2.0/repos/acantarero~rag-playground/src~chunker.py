from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.state import State

class Chunker:
    # light wrapper for chunking methods

    def __init__(self, state: State):
        self.state = state

    def chunk(self, chunk_size: int, overlap: int) -> list[str]:
        """Chunk document.
        
        Store chunks in applicaiton state.
        Return chunks for UI display.
        """
        # gradio number field is a float
        chunk_size = int(chunk_size)
        overlap = int(overlap)  

        text = self.state.get_document_text()
        if text is None:
            raise ValueError("No document text loaded. Use documents tab to load a document.")

        if self.state.get_chunking_method() == "recursive_character":
            self.state.set_chunks(self._recursive_character_splitter(text, chunk_size, overlap))
            # format for UI display
            return "\n\n".join(["Chunk:\n" + _ for _ in self.state.get_chunks()])
        else:
            raise ValueError(f"Invalid chunking method. Set on chunking tab.")

    def _recursive_character_splitter(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        return text_splitter.split_text(self.state.get_document_text())
