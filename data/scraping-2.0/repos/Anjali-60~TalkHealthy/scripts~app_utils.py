import logging
import math
import os
import platform
import sys
import textwrap
from typing import List

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    JSONLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
)
from langchain.schema import Document

from scripts.app_environment import cli_column_number, cli_column_width


def print_platform_version():
    """
    The sys.platform for macOS is 'darwin', for Windows it's 'win32', and for Linux it's 'linux'
    (it can be more specific like 'linux2' or 'linux3', depending on the Linux version you're running).
    The platform.machine() returns the machine type, like 'x86_64' or 'amd64' for an Intel x64 machine, and 'arm64' for an ARM64 machine.
    """
    logging.debug("sys_platform:", sys.platform)
    logging.debug("platform_machine:", platform.machine())


######################################################################
# INGEST
######################################################################

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fall back to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8", "autodetect_encoding": True}),
    ".json": (JSONLoader, {"jq_schema": ".[].full_text"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    """
    The function takes a single file and loads its data using the appropriate loader based on its extension.
    :param file_path: The path of the file to load.
    :return: A list of Document objects loaded from the file.
    """
    ext = (os.path.splitext(file_path)[-1]).lower()
    if ext in LOADER_MAPPING:
        try:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()
        except Exception as e:
            raise ValueError(f"Problem with document {file_path}: \n'{e}'")
    raise ValueError(f"Unsupported file extension '{ext}'")


######################################################################
# DISPLAY
######################################################################

def display_source_directories(folder: str) -> list[str]:
    """
    Displays the list of existing directories in the folder directory.
    :return: The list of existing directories.
    """
    print(f"\nExisting directories in ./{folder}:")
    return sorted((f for f in os.listdir(f"./{folder}") if not f.startswith(".")), key=str.lower)


def display_directories():
    """
    This function displays the list of existing directories in the parent directory.
    It also explores one level of subdirectories for each directory.
    :return: The list of existing directories.
    """
    base_dir = "source_documents"
    directories = display_source_directories(base_dir)

    # Flat list to keep track of all directories and subdirectories
    all_dirs = []

    for directory in directories:
        # Get the subdirectories for the current directory
        sub_directories = os.scandir(os.path.join(base_dir, directory))
        # Only keep the subdirectories (not files)
        sub_directories = [f"{directory}/{sd.name}" for sd in sub_directories if sd.is_dir()]

        # Add directory and its subdirectories to the list
        all_dirs.append(directory)
        all_dirs.extend(sub_directories)

    # Calculate the number of rows needed based on the number of directories
    num_rows = math.ceil(len(all_dirs) / cli_column_number)

    # Print directories in multiple columns
    for row in range(num_rows):
        for column in range(cli_column_number):
            # Calculate the index of the directory based on the current row and column
            index = row + column * num_rows

            if index < len(all_dirs):
                directory = all_dirs[index]
                wrapped_directory = textwrap.shorten(directory, width=cli_column_width - 1, placeholder="...")
                print(f"{index + 1:2d}. {wrapped_directory:{cli_column_width}}", end="")
        print()  # Print a new line after each row

    return all_dirs
