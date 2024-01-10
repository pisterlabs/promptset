from langchain.document_loaders import DirectoryLoader
from llama_index import SimpleDirectoryReader
from llama_index import Document

document = Document(
    text='text',
    metadata={
        'filename': '<document_id>',
        'category' : '<category>'
    }
)

document.metadata = {'filename': '<doc_file_name>'}


documents = SimpleDirectoryReader('/Users/dmg/Desktop/coding /RaG/data', filename_as_id=True, file_metadata='').load_data()

print(documents[0].dict)



#  """Simple directory reader.

#     Load files from file directory.
#     Automatically select the best file reader given file extensions.

#     Args:
#         input_dir (str): Path to the directory.
#         input_files (List): List of file paths to read
#             (Optional; overrides input_dir, exclude)
#         exclude (List): glob of python file paths to exclude (Optional)
#         exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
#         encoding (str): Encoding of the files.
#             Default is utf-8.
#         errors (str): how encoding and decoding errors are to be handled,
#               see https://docs.python.org/3/library/functions.html#open
#         recursive (bool): Whether to recursively search in subdirectories.
#             False by default.
#         filename_as_id (bool): Whether to use the filename as the document id.
#             False by default.
#         required_exts (Optional[List[str]]): List of required extensions.
#             Default is None.
#         file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
#             extension to a BaseReader class that specifies how to convert that file
#             to text. If not specified, use default from DEFAULT_FILE_READER_CLS.
#         num_files_limit (Optional[int]): Maximum number of files to read.
#             Default is None.
#         file_metadata (Optional[Callable[str, Dict]]): A function that takes
#             in a filename and returns a Dict of metadata for the Document.
#             Default is None.
#     """