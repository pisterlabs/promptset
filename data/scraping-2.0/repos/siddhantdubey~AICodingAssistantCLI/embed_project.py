import ast
import os
import chromadb
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

client = chromadb.PersistentClient(path="/mnt/d/Documents/Projects/code-assistant-cli/embeddings")


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def read_code_from_file(filename):
    with open(filename, "r") as file:
        code = file.read()
    return code


def extract_functions_and_classes(code):
    module = ast.parse(code)
    functions = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            parent_class = [n for n in ast.walk(node) if isinstance(n, ast.ClassDef)]
            parent_class_name = parent_class[0].name if parent_class else None
            functions.append((node, parent_class_name))
    return functions


def function_code(function, code):
    lines = code.split('\n')
    return '\n'.join(lines[function.lineno - 1: function.end_lineno])


def process_code(code, filename):
    functions_and_classes = extract_functions_and_classes(code)
    function_names_to_code = {}
    embeddings = {}
    for function, class_name in functions_and_classes:
        func_code = function_code(function, code)
        full_name = f"{filename}_{class_name}_{function.name}" if class_name else f"{filename}_{function.name}"
        function_names_to_code[full_name] = func_code
        embeddings[full_name] = get_embedding(func_code)
    return embeddings, function_names_to_code


def add_to_chromadb(collection, embeddings, sources):
    embeddings_list = [embeddings[func_name] for func_name in embeddings]
    function_bodies = [sources[function_name] for function_name in sources]
    collection.add(
        embeddings=embeddings_list,
        documents=function_bodies,
        metadatas=[{"source": function_name} for function_name in sources],
        ids=[function_name for function_name in embeddings]
    )

