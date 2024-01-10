import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema.document import Document

class CodeSplitter:
    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 200, ignore_folders: Optional[List[str]] = None, ignore_files: Optional[List[str]] = None):
        """
        Initialize a CodeSplitter instance.

        Args:
            chunk_size (int): The size of code chunks to split.
            chunk_overlap (int): The overlap between code chunks.
            ignore_folders (Optional[List[str]]): A list of folders to ignore when processing a directory.
            ignore_files (Optional[List[str]]): A list of files to ignore when processing a directory.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ignore_folders = ignore_folders or []
        self.ignore_files = ignore_files or []

    def _split_code(self, code: str, language: Language) -> List[Document]:
        """
        Split code into a list of documents based on the provided language.

        Args:
            code (str): The code to split.
            language (Language): The language of the code.

        Returns:
            List[Document]: A list of Document objects representing code segments.
        """
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            documents = splitter.create_documents([code])
            return documents
        except Exception as e:
            raise RuntimeError(f"Error splitting code: {str(e)}")

    def _check_file_language(self, file_extension: str) -> Optional[Language]:
        """
        Check the language of a file based on its extension.

        Args:
            file_extension (str): The file extension to check.

        Returns:
            Optional[Language]: The Language enum corresponding to the file extension, or None if not recognized.
        """
        file_extension = file_extension.lower()
        extension_to_language = {
            'cpp': Language.CPP,
            'go': Language.GO,
            'java': Language.JAVA,
            'kt': Language.KOTLIN,
            'js': Language.JS,
            'ts': Language.TS,
            'php': Language.PHP,
            'proto': Language.PROTO,
            'py': Language.PYTHON,
            'rst': Language.RST,
            'rb': Language.RUBY,
            'rs': Language.RUST,
            'scala': Language.SCALA,
            'swift': Language.SWIFT,
            'md': Language.MARKDOWN,
            'tex': Language.LATEX,
            'html': Language.HTML,
            'sol': Language.SOL,
            'cs': Language.CSHARP,
        }
        return extension_to_language.get(file_extension, None)

    def _read_file(self, file_path: str) -> str:
        """
        Read the content of a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _process_directory(self, directory_path: str) -> List[Document]:
        """
        Process a directory, reading and splitting code files.

        Args:
            directory_path (str): The path to the directory.

        Returns:
            List[Document]: A list of Document objects representing code segments from the directory.
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path")

        code_documents = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # Check if the folder or file should be ignored
            if (os.path.isdir(file_path) and filename in self.ignore_folders) or (os.path.isfile(file_path) and filename in self.ignore_files):
                continue

            if os.path.isfile(file_path):
                file_extension = os.path.splitext(filename)[1].lstrip(".").lower()
                language = self._check_file_language(file_extension)

                if language:
                    code = self._read_file(file_path)
                    code_segments = self._split_code(code, language)
                    code_documents.extend(code_segments)

        return code_documents
    
    def __call__(self, directory_path: str) -> List[Document]:
        """
        Process a directory and return a list of code segments.

        Args:
            directory_path (str): The path to the directory.

        Returns:
            List[Document]: A list of Document objects representing code segments from the directory.
        """
        docs = self._process_directory(directory_path)
        return docs
