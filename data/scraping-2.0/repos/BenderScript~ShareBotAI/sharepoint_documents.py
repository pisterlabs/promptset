import os
import mimetypes

from langchain.text_splitter import RecursiveCharacterTextSplitter
from office365.sharepoint.client_context import ClientContext

from file_type import FileType, file_handler_map


def download_files(source_folder, download_path):
    """
    Enumerate folder files and download file's content

    :type source_folder: Folder
    :type download_path: str
    """

    # 1. retrieve files collection (metadata) from library root folder
    files = source_folder.files.get().execute_query()
    downloaded_files = []

    # 2. start download process (per file)
    for file in files:
        print("Downloading file: {0} ...".format(file.properties["ServerRelativeUrl"]))
        download_file_name = os.path.join(download_path, file.name)
        with open(download_file_name, "wb") as local_file:
            file.download(local_file).execute_query()
        print(f"[Ok] file has been downloaded: {download_file_name}")
        downloaded_files.append(download_file_name)
    return downloaded_files


def get_file_type(file_path):
    """Uses the file extension to determine the file type"""
    # Check if file exists
    if not os.path.exists(file_path):
        return "File does not exist"

    _, ext = os.path.splitext(file_path)
    ext = ext[1:]  # Remove the leading '.'

    try:
        file_type = FileType(ext).name
    except ValueError:
        file_type = "Unknown file type"

    return file_type


# Define a function to load a document based on its file type


def split_text_docs(session_state):
    # Split docs
    # TODO: The chunk size needs to be chosen such that the number of texts is less than 16 for Azure AI
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0, length_function=len,
                                                   is_separator_regex=False, )
    # TODO: Try SpacySentenceTextSplitter
    texts = text_splitter.split_documents(session_state.loaded_docs)
    return texts


def load_document(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext[1:]  # Remove the leading '.'
    try:
        file_type = FileType(ext)
    except ValueError:
        print(f"Unsupported file type: {ext}")
        return None

    # Call the handling function for this file type
    loader = file_handler_map[file_type](file_path)

    return loader.load()


# Traverse a directory and load documents
def traverse_directory(directory_path):
    documents = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            document = load_document(file_path)
            if document is not None:
                documents.append(document[0])

    return documents


def determine_file_type(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        return "File does not exist"

    # Get the MIME type of the file
    mime_type = mimetypes.guess_type(file_path)[0]

    # Determine file type based on MIME type
    if mime_type is not None:
        if "pdf" in mime_type:
            return "PDF file"
        elif "powerpoint" in mime_type or "officedocument.presentation" in mime_type:
            return "PowerPoint file"
        elif "word" in mime_type or "officedocument.wordprocessing" in mime_type:
            return "PowerPoint file"
        elif "mp4" in mime_type:
            return "MP4 file"

    return "Unknown file type"


def connect_to_sharepoint(session_state):
    site_url = session_state.sharepoint_url
    client_id = os.getenv("OFFICE365_CLIENT_ID")
    client_secret = os.getenv("OFFICE365_CLIENT_SECRET")

    ctx = ClientContext(site_url).with_client_credentials(
        client_id, client_secret
    )

    folder_relative_url = session_state.sharepoint_folder
    folder = ctx.web.get_folder_by_server_relative_url(folder_relative_url).select(["Exists"]).get().execute_query()
    if folder.exists:
        print("Folder '{0}' is found".format(folder_relative_url))
        return folder
    else:
        print("Folder '{0}' not found".format(folder_relative_url))
        return False
