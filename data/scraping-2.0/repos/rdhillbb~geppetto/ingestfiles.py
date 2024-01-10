import os
import glob
import shutil
from typing import Optional, List
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from ganamation import startanamation, stopanamation
import validators

def is_valid_download_url(url)-> bool:
    # Parse the URL
    parsed_url = urlparse(url)

    # Check if the scheme is valid for downloading (http, https, ftp, file, etc.)
    valid_schemes = ("http", "https", "ftp", "file")  # Add more schemes if needed
    if parsed_url.scheme and parsed_url.scheme.lower() in valid_schemes:
        return True 
    
    return False


def download_file(url, user_agent="Mozilla") -> str:
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, headers=headers, stream=True)
        status = response.raise_for_status()
        print(status)

        # Extract the filename from the URL
        file_name = os.path.basename(urlparse(url).path)

        # Create the "working" directory if it doesn't exist
        directory = "working"
        directory = os.path.join(os.environ['GEPPETTO_FILE_CABINET'], 'bottomDrawer')
        os.makedirs(directory, exist_ok=True)

        # Save the file in the "working" directory
        file_path = os.path.join(directory, file_name)
        total_size = int(response.headers.get("Content-Length", 0))

        with open(file_path, "wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=file_name, ncols=80
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))

        print("File downloaded successfully.")
        return os.path.abspath(file_path)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during file download: {str(e)}")
        return None




"""
    Lists files with specific extensions in a directory.

    Args:
        directory (str): The directory path.
        extensions (List[str], optional): List of file extensions. Defaults to common extensions.
     Note: extension list is not provided.
    Returns:
        List[str]: List of absolute file paths.
    """
def list_files_with_ext(directory, extensions=None) -> List[str]:
    if extensions is None:
        extensions = ['txt', 'pdf', 'docx', 'csv', 'md', 'html','ppt']
    files = []
    for extension in extensions:
        files.extend(glob.glob(directory + f'/**/*.{extension}', recursive=True))
    return [os.path.abspath(file) for file in files]



"""
    Copies a file to a new directory while preserving its filename and directory structure.

    Args:
        file_path (str): The path of the file to be copied.

    Returns:
        str: The path of the newly copied file.
    """
def copy_file_to_new_dir(file_path)-> str:
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    new_dir = os.path.join(os.environ['GEPPETTO_FILE_CABINET'], 'middledrawer', file_name)

    os.makedirs(new_dir, exist_ok=True)

    new_file_path = os.path.join(new_dir, base_name)
    shutil.copy2(file_path, new_file_path)
    return new_file_path

def ing_singlefile(file_path, chunksize: int = 1000, overlap: int = 300) -> Optional[str]:
    try:
        file = copy_file_to_new_dir(file_path)
        file_drawer = os.path.dirname(file)
        persist_directory = os.path.join(file_drawer, "fiass_index")
        startanamation('Ingesting ..... '+file)      
        embedding = OpenAIEmbeddings()
        loader = UnstructuredFileLoader(file, strategy="fast", mode="elements")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=overlap)
        docs = text_splitter.split_documents(documents)
        
        
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(persist_directory)
        stopanamation(True) 
        return "Complete"
    except Exception as e:
        stopanamation(True) 
        return f"ERROR: {str(e)}"

     
def ing_dirconsolidate(dir_path: str, chunksize: int = 1000, overlap: int = 300) -> str:
    try:
        if os.path.exists(dir_path):
            basename = os.path.basename(dir_path)
        else:
            return "ERROR: Directory DOES NOT EXIST"
        new_dir = os.path.join(os.environ['GEPPETTO_FILE_CABINET'], 'middleDrawer', basename)
        startanamation("Ingesting ..... Files in "+new_dir) 
        
        # Check if the destination directory exists
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        
        for filename in os.listdir(dir_path):
            source_file = os.path.join(dir_path, filename)
            destination_file = os.path.join(new_dir, filename)
            shutil.copy2(source_file, destination_file)
        
        persist_directory = f"{new_dir}/fiass_index"
        loader = DirectoryLoader(new_dir, glob='./*.*')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=overlap)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(persist_directory)
        stopanamation(True) 
        return "Complete"
    except Exception as e:
        stopanamation(True) 
        return f"ERROR: {str(e)}"

def ing_httpfile(url:str,chunksize: int = 1000, overlap: int = 300) -> str:
     dlfile = download_file(url)
     return ing_singlefile(dlfile) 

def ing_filesFrmDr(directory, chunksize: int = 1000, overlap: int = 300,extensions=None):
    if extensions is None:
        extensions = ['txt', 'pdf', 'docx', 'csv', 'md', 'html']
    
    files = list_files_with_ext(directory, extensions)
    for file in files:
        print("\nLoading", file)
        try:
            ing_singlefile(file)
        except Exception as e:
            print(f"An error occurred with file {file}: {e}")
            continue
# Use the function
def ask_user_intypedir():
    while True:
        response = input("Do you want to consolidate all files in the directory into a single repository or separate? Type 'Y' for consolidate, 'N' for separate repository: ")
        if response == ".done":
            return response
        if response.lower() in ['y', 'yes', 'n', 'no']:
            return response.lower() in ['y', 'yes']
        else:
            print("Invalid input. Please type 'Y', 'Yes', 'N' or 'No'.")
            continue 
# Use the function

def userinput(s:str )-> str:
    valid_extensions = ['txt', 'pdf', 'docx', 'csv', 'md', 'html', 'ppt']
    if validators.url(s):
        path = urlparse(s).path
        if '.' in path and path.split('.')[-1] in valid_extensions:
            ing_httpfile(s) 
        else:
            print("URL must have a file end point of "+ valid_extensions)

    elif os.path.isdir(s):
          rsp = ask_user_intypedir()
          if rsp == ".done":
              print("No directory Ingestion")
          else:
              if rsp:
                  ing_dirconsolidate(s)
              else:
                  ing_filesFrmDr(s)
    elif os.path.isfile(s):
        ing_singlefile(s) 
    else:
        print("Unknown")

while True:
    rsp = input("Input the item you wish to inest into the file cabinate? \"type .done\" to exit\n")
    if rsp == ".done":
        print("Exiting Ingestion Service")
        exit()
    userinput(rsp) 
print("Exiting Injestion")    
