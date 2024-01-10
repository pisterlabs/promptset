from git import Repo
from langchain.document_loaders import GitLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
import os
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, Document
from langchain.prompts import PromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from dotenv import load_dotenv
import json
import time
from pydantic import BaseModel, Field
import subprocess
from thefuzz import process, fuzz

def find_closest_file(directory, filepath, threshold=95):
    files = list_non_ignored_files(directory)
    closest_match = process.extractOne(filepath, files, scorer=fuzz.token_sort_ratio)
    print("find_closest_file: closest_match: ", closest_match)
    if closest_match[1] < threshold:
        return filepath
    else:
        print("Found closest file in find_closest_file: ", directory, filepath, closest_match[0])
        return closest_match[0]

def list_non_ignored_files(directory):
    code_suffixes = [".py", ".js", ".jsx", ".tsx", ".ts", ".cc", ".hpp", ".cpp", ".c", ".rb"] # make a better list
    find_command = f"cd {directory} && git ls-files --exclude-standard && git ls-files --exclude-standard -o"
    result = subprocess.run(find_command, shell=True, check=True, text=True, capture_output=True)
    non_ignored_files = result.stdout.splitlines()

    print("Found non_ignored_files output: ", non_ignored_files)

    # suffix_non_ignored_files = [] - BUGGY
    # for filepath in non_ignored_files:
    #     ext = filepath.split(".")[-1]
    #     if ext in code_suffixes:
    #         suffix_non_ignored_files.append(filepath)

    # print("Found non_ignored_files: ", suffix_non_ignored_files)

    return non_ignored_files

def get_documents(directory, ignore_file=".gitignore", no_gitignore=False, parser_threshold=1000):
    files = list_non_ignored_files(directory)
    code_suffixes = ["py", "js", "jsx", "tsx", "ts", "cc", "hpp", "cpp", "rb"] # make a better list
    language_map = {
        "py": Language.PYTHON,
        "js": Language.JS,
        "java": Language.JAVA,
        "ts": Language.TS,
        "tsx": Language.TS,
        "js": Language.JS,
        "jsx": Language.JS,
        "cc": Language.CPP,
        "hpp": Language.CPP,
        "cpp": Language.CPP,
        "rb": Language.RUBY
    }

    all_docs = []

    for rfilepath in files:
        ext = rfilepath.split(".")[-1]
        if ext in code_suffixes:
            filepath = os.path.join(directory, rfilepath)
            print("Loading: ", filepath)
            file = open(filepath, "r")
            contents = file.read()
            file.close()      

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_map[ext],
                chunk_size=250,
                chunk_overlap=20
            )

            original_doc = Document(
                page_content=contents,
                metadata={
                    "source": filepath,
                    "last_modified": time.time()
                }
            )

            docs = splitter.split_documents([original_doc])

            all_docs.extend(docs)
            print("Finished.")
   
    return all_docs

if __name__ == "__main__":
    print("Main")
    print(get_documents("/Users/vijaydaita/Files/uiuc/rxassist/rxassist/"))