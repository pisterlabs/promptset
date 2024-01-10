import os
import json
import openai
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
from knowledge_base import load_documents, supabase_vdb, local_vdb, load_local_vdb
from collections import deque
from pathlib import Path
import util
import subprocess
import gradio as gr
import cohere
# import torch
# from run import model, tokenizer
# from llama_index.indices.query.schema import QueryBundle, QueryType
# from llama_index.schema import NodeWithScore
# from llama_index.indices.postprocessor import SentenceTransformerRerank
# from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
co = cohere.Client("VqlHIhTs1QqHQM3nNkt18t0K5sVJQ3ykcqBW0Psz")

def clone_repo(git_url, progress=gr.Progress(), code_repo_path="./code_repo"):
    print(progress(0.1, desc="Cloning the repo..."))
    print("Cloning the repo: ", git_url)
    # Check if directory exists
    if not os.path.exists(code_repo_path):
        os.makedirs(code_repo_path)
    try:
        subprocess.check_call(['git', 'clone', git_url], cwd=code_repo_path)
        print(f"Successfully cloned {git_url} into {code_repo_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")

    print(progress(0.3, desc="Summarizing the repo..."))
    # readme_info = get_readme(code_repo_path)
    # if readme_info is not None:
    #     readme_info = """The README.md file is as follows: """ + readme_info + "\n\n"

    print(progress(0.4, desc="Parsing repo structure..."))
    repo_structure = get_repo_structure(code_repo_path)
    if repo_structure is not None:
        repo_structure = """The repo structure is as follows: """ + get_repo_structure(code_repo_path) + "\n\n"

    return repo_structure


def generate_knowledge_from_repo(dir_path, ignore_list):
    print("Ignore list: ", ignore_list)
    knowledge = {"known_docs": [], "known_text": {"pages": [], "metadatas": []}}
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in ignore_list]  # modify dirs in-place
        for file in files:
            if file.endswith(tuple(ignore_list)):
                filepath = os.path.join(root, file)
                try:
                    knowledge["known_docs"].extend(load_documents([filepath]))
                except Exception as e:
                    print(f"Failed to process {filepath} due to error: {str(e)}")
    print("TOTAL DOCS - ", len(knowledge["known_docs"]))
    return knowledge


# Find the Readme.md file from the code repo in the code_repo folder
def find_repo_folder(directory):
    # Find the name of the folder in the specified directory
    folder_name = None
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_name = item
            break
    return os.path.join(directory, folder_name)


def find_readme(repo_folder):
    # Search for the README file within the found folder
    for filename in os.listdir(repo_folder):
        if filename.lower().startswith('readme'):
            readme_path = os.path.join(repo_folder, filename)
            print("README found in folder:", repo_folder)
            return readme_path

    print("README not found in folder:", repo_folder)
    return None

def bfs_folder_search(text_length_limit=40000, folder_path="./code_repo"):
    if not Path(folder_path).is_dir():
        return "Invalid directory path"

    root = Path(folder_path).resolve()
    file_structure = {str(root): {}}
    queue = deque([(root, file_structure[str(root)])])

    while queue:
        current_dir, parent_node = queue.popleft()
        try:
            for path in current_dir.iterdir():
                if path.is_dir():
                    if str(path.name) == ".git":
                        continue
                    parent_node[str(path.name)] = {"files": []}
                    queue.append((path, parent_node[str(path.name)]))
                else:
                    if "files" not in parent_node:
                        parent_node["files"] = []
                    parent_node["files"].append(str(path.name))

                # Check if we've exceeded the text length limit
                file_structure_text = json.dumps(file_structure)
                if len(file_structure_text) >= text_length_limit:
                    return file_structure_text

        except PermissionError:
            # This can happen in directories the user doesn't have permission to read.
            continue

    return json.dumps(file_structure)


def get_repo_structure(code_repo_path="./code_repo"):
    return bfs_folder_search(4000, code_repo_path)


def get_repo_names(dir_path):
    folder_names = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    concatenated_names = "-".join(folder_names)
    return concatenated_names


def generate_or_load_knowledge_from_repo(dir_path="./code_repo"):
    vdb_path = "./vdb-" + get_repo_names(dir_path) + ".pkl"
    # check if vdb_path exists
    if os.path.isfile(vdb_path):
        print(colored("Local VDB found! Loading VDB from file...", "green"))
        vdb = load_local_vdb(vdb_path)
    else:
        print(colored("Generating VDB from repo...", "green"))
        ignore_list = [".swift", ".md", ".ts", ".py"]
        knowledge = generate_knowledge_from_repo(dir_path, ignore_list)
        vdb = local_vdb(knowledge, vdb_path=vdb_path)
    print(colored("VDB generated!", "green"))
    return vdb

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def get_repo_context(query, vdb):
    matched_docs = vdb.similarity_search(query, k=25)
    # print("BEFORE RERANKING: ")
    # pretty_print_docs(matched_docs)
    docs = []
    for doc in matched_docs:
        docs.append(doc) 
    f = [doc.page_content for doc in docs]
    # with torch.no_grad():
    #     inputs = tokenizer(all_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    #     scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    # indices = torch.argsort(scores, descending=True)[:10]
    # matched_docs = [matched_docs[idx] for idx in indices]
    results_rerank = co.rerank(query=query, documents = f, top_n=10, model="rerank-english-v2.0")
    rerank_docs = [result.document for result in results_rerank.results]
    # print("RERANKED DOCS:")

    output = ""
    for idx, docs in enumerate(rerank_docs):
        output += f"Context {idx}:\n"
        output += str(docs)
        # print("DOCUMENT ", idx)
        # print(docs)
        # print("-----------------------------------------------------------------")
        output += "\n\n"
    return output


if __name__ == '__main__':
    code_repo_path = "./code_repo"
    load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY", "null")

    print(get_repo_names(code_repo_path))

    # Basic repo information
    # get_readme(code_repo_path)
    print(colored(bfs_folder_search(4000, code_repo_path), "yellow"))

    # Generate knowledge base
    vdb = generate_or_load_knowledge_from_repo("./code_repo")

    # Search the knowledge base
    query = "How to use the knowledge base?"
    context = get_repo_context(query, vdb)
    print(context)
