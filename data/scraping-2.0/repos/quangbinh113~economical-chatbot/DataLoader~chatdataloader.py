import os
from typing import List, Any

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import JSONLoader
from pathlib import Path
import json
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.text import TextLoader
import zipfile
import rarfile
import json
from pathlib import Path
from langchain.document_loaders import PyPDFLoader


class FileLoader(object):
    """
    Class load file
    -> User will be able to load a file with an abytrary extension
    -> The file will be loaded using the appropriate loader
    -> The output will be a list of documents
    """

    def __init__(self) -> None:
        """
        Class Attributes:
            csv_data: (list) -> A list of file contents from the csv.
            pdf_data: (list) -> A list of file contents from the pdf.
            markdown_data: (list) -> A list of file contents from the markdown.
            directory_data: (list) -> A list of file contents from the directory.
            json_data: (list) -> A list of file contents from the json.
            text: (list) -> A list of file contents from the text.
        """
        self.csv_data = None
        self.pdf_data = None
        self.markdown_data = None
        self.directory_data = None
        self.json_data = None
        self.text = None
        # self.extension = None

    def get_file_extension(self, filename: str):
        """
        Func return .name file
        Args:
            filename: (str) -> The filename to get the extension of.
        Returns:
            str: The extension of the filename.
        """
        return os.path.splitext(filename)[1]

    def text_loader(self, text_file: str) -> list:
        """
        Load and processing text file -> return documents
        Args:
            text_file: (str) -> The text file to load.
        Returns:
            List[str]: A list of file contents from the text.
        """
        loader = TextLoader(text_file)
        self.text = loader.load()
        return self.text

    def csv_loader(self, csv_file: str) -> list:
        """
        Load và processing CSV by Langchain -> return documents
        Args:
            csv_file: (str) -> The csv file to load.
        Returns:
            List[str]: A list of file contents from the csv.
        """
        loader = CSVLoader(
            file_path=csv_file,
            encoding="utf-8",
            csv_args={
                "delimiter": ",",
            },
        )
        self.csv_data = loader.load()

        return self.csv_data

    def pdf_loader(self, pdf_file: str) -> list:
        """
        Get text of file PDF -> return documents
        Args:
            pdf_file: (str) -> The pdf file to load.
        Returns:
            List[str]: A list of file contents from the pdf.
        """
        loader = PyPDFLoader(pdf_file)
        self.pdf_data = loader.load()
        return self.pdf_data

    def markdown_loader(self, markdown_file: str) -> list:
        """
        Get text of file markdown -> return documents
        Args:
            markdown_file: (str) -> The markdown file to load.
        Returns:
            List[str]: A list of file contents from the markdown.
        """
        markdown_loader = UnstructuredMarkdownLoader(markdown_file)
        self.markdown_data = markdown_loader.load()
        return self.markdown_data

    def json_loader(self, json_file_path: str) -> str:
        """
        Get list text file Json -> return documents
        Args:
            json_file_path: (str) -> The json file path to load.
        Returns:
            List[str]: A list of file contents from the json.
        """
        self.json_data = json.loads(Path(json_file_path).read_text())
        return self.json_data

    def directory_loader(self, directory_file: str) -> list:
        """
        Check directory file -> .zip or rar unzip and check types of file in directory
        Args:
            directory_file: (str) -> The directory file to load.
        Returns:
            List[str]: A list of file contents from the directory.
        """
        # Lấy danh sách tệp trong thư mục và đọc nội dung từ mỗi tệp
        loaders = {
            ".pdf": PyMuPDFLoader,
            ".xml": UnstructuredXMLLoader,
            ".csv": CSVLoader,
            ".json": JSONLoader,
            ".md": UnstructuredMarkdownLoader,
            ".txt": TextLoader,
        }

        def _load_archive_contents(directory_file):
            """
            Load the contents of an archive file (either .zip or .rar).
            Args:
                archive_file_path (str): The path to the archive file.
            Returns:
                List[str]: A list of file contents from the archive.
            """
            try:
                archive_documents = []
                if directory_file.endswith(".zip"):
                    # Handle .zip files
                    with zipfile.ZipFile(directory_file, "r") as zip_file:
                        for inner_filename in zip_file.namelist():
                            doc = _process_directory(inner_filename)
                            # with zip_file.open(inner_filename) as inner_file:
                            #     # Read and decode the content as utf-8
                            #     content = inner_file.read().decode('utf-8')
                            archive_documents.append(doc)
                elif directory_file.endswith(".rar"):
                    # Handle .rar files
                    with rarfile.RarFile(directory_file, "r") as rar_file:
                        for inner_filename in rar_file.namelist():
                            doc = _process_directory(inner_filename)
                            # with rar_file.open(inner_filename) as inner_file:
                            #     # Read and decode the content as utf-8
                            #     content = inner_file.read().decode('utf-8')
                            archive_documents.append(doc)
                return archive_documents
            except Exception as e:
                print(f"Error loading contents from {directory_file}: {e}")
                return []

        def _create_directory_loader(file_type, directory_file):
            return DirectoryLoader(
                path=directory_file,
                glob=f"**/*{file_type}",
                loader_cls=loaders[file_type],
            )

        def _process_directory(directory_file):
            documents = {}
            for file_type, loader_cls in loaders.items():
                loader = _create_directory_loader(file_type, directory_file, loader_cls)
                documents[file_type] = loader.load()

            return documents

        self.directory_data = _load_archive_contents(directory_file)
        return self.directory_data

    def load_file(self, file_path: str) -> list | str | list[Any]:
        """
        Load a file and determine its type, then call the appropriate loader.
        Args:
            file_path: (str) -> The path to the file to load.
        Returns:
            List[str]: A list of file contents.
        """
        file_extension = self.get_file_extension(file_path)

        try:
            if file_extension == ".pdf":
                return self.pdf_loader(file_path)
            elif file_extension == ".csv":
                return self.csv_loader(file_path)
            elif file_extension == ".md":
                return self.markdown_loader(file_path)
            elif file_extension == ".json":
                return self.json_loader(file_path)
            elif file_extension == ".txt":
                return self.text_loader(file_path)
            elif file_extension in (".zip", ".rar"):
                return self.directory_loader(file_path)
            else:
                print(f"Unsupported file type: {file_extension}")
                return []
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []


# if __name__ == "__main__":
#     # load = FileLoader().csv_loader(r'C:\Users\binh.truong\Code\economical-chatbot\file_upload\VIC.csv')
#     # load2 = FileLoader().json_loader(r'C:\Users\binh.truong\Code\economical-chatbot\file_upload\tesst.json')
#     load3 = FileLoader()
#     x = load3.load_file(r"C:\Users\anh.do\Desktop\economical-chatbot\NLP with hugging face.pdf")
#     print(x[330])

    # print(load3)