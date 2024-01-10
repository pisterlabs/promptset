import tree_sitter_languages

from typing import Any, List
from langchain.text_splitter import TextSplitter

class CodeSplitter(TextSplitter):
    """
    CodeSplitter is designed to split code into chunks by parsing it into an
    Abstract Syntax Tree (AST) using the tree_sitter_languages library.
    It takes in language, chunk lines, chunk lines overlap, and maximum characters
    to create manageable pieces of code.
    """

    def __init__(self, language: str, chunk_size: int = 40,
                 chunk_overlap: int = 15, max_chars: int = 1500):
         self.language = language
         self.chunk_size = chunk_size
         self.chunk_overlap = chunk_overlap
         self.max_chars = max_chars

    def _chunk_node(self, node: Any, text: str, last_end: int = 0) -> List[str]:
        new_chunks = []
        current_chunk = ""
        for child in node.children:
            # Check if child node exceeds maximum characters
            if child.end_byte - child.start_byte > self.max_chars:
                # If current chunk has content, add it to the new chunks
                if current_chunk:
                    new_chunks.append(current_chunk)
                current_chunk = ""
                # Recursively chunk child node
                new_chunks.extend(self._chunk_node(child, text, last_end))
            elif (
                len(current_chunk) + child.end_byte - child.start_byte > self.max_chars
            ):
                # Add current chunk to new chunks if next child exceeds max characters
                new_chunks.append(current_chunk)
                current_chunk = text[last_end : child.end_byte]
            else:
                # Append text to the current chunk
                current_chunk += text[last_end : child.end_byte]
            last_end = child.end_byte

        # Append remaining current chunk if not empty
        if current_chunk:
            new_chunks.append(current_chunk)
        return new_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Splits incoming code and return chunks using the AST.
        Incorporates error handling for unsupported languages or missing dependencies.
        """
        try:
            parser = tree_sitter_languages.get_parser(self.language)
        except Exception as e:
            error_message = (
                f"Could not get parser for language {self.language}."
            )
            print(error_message)
            raise e

        tree = parser.parse(bytes(text, "utf-8"))

        # Validate tree and parse code into chunks
        if not tree.root_node.children or tree.root_node.children[0].type != "ERROR":
            chunks = [chunk.strip() for chunk in self._chunk_node(tree.root_node, text)]
            return chunks
        else:
            raise ValueError(f"Could not parse code with language {self.language}.")
