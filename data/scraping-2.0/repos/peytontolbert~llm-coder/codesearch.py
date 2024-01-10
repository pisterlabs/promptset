import requests
import os
from dotenv import load_dotenv
import tiktoken
from glob import glob
import openai
import pandas as pd

embeddinglimit = 8190
tokenlimit = 2047
input_datapath = "data/test.csv"
df = pd.DataFrame(all_funcs)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']

def split_into_chunks (text, limit, overlap, encmodel):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    
    for line in lines:
        tokens_in_line = num_tokens_from_string(line, encmodel)
        if current_length + tokens_in_line > limit:
            # Create a new chunk
            chunks.append('\n'.join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = sum(num_tokens_from_string(line, encmodel) for line in current_chunk)

        current_chunk.append(line)
        current_length += tokens_in_line

    # Add the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

def get_function_name(code):
    """
    Extract function name from a line beginning with "def "
    """
    assert code.startwith("def ")
    return code[len("def "): code.index("(")]

def get_until_no_space(all_lines, i) -> str:
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, i + 10000):
        if j< len(all_lines):
            if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
                ret.append(all_lines[j])
            else:
                break
    return "\n".join(ret)

def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    whole_code = open(filepath).read().replace("\r", "\n")
    all_lines = whole_code.split("\n")
    for i, l in enumerate(all_lines):
        if l.startswith("def "):
            code = get_until_no_space(all_lines, i)
            function_name = get_function_name(code)
            yield {"code": code, "function_name": function_name, "filepath": filepath}
                      

def get_file_paths(owner, repo):
    # Github API url
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"

    # Make the API request
    response = requests.get(url)

    # Get the JSON data from the response
    data = response.json()

    # Initialize an empty list for the file paths
    file_paths = []

    # Loop through each file in the data
    for file in data["tree"]:
        # If the file is not a directory, add its path to the list
        if file["type"] != "tree":
            file_paths.append(file["path"])
    
    return file_paths

def main():
    load_dotenv()
    owner = os.getenv('REPO_OWNER')
    repo = os.getenv('REPO_NAME')
    text = "Hello world how many tokens"
    encmodel = "cl100k_base"
    tokens = num_tokens_from_string(text, encmodel)
    print(tokens)
    df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model="text-embedding-ada-002"))
    df.to_csv('output/embeddings.csv')
    file_paths = get_file_paths(owner, repo, encmodel)
    print(*file_paths, sep='\n')

if __name__ == "__main__":
    main()