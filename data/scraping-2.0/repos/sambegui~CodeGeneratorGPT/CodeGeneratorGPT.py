import os
import re
import openai
from pathlib import Path
import dotenv
import spacy
import shutil

dotenv.load_dotenv()
nlp = spacy.load("en_core_web_sm")

def count_tokens(text):
    doc = nlp(text)
    return len(doc)

def generate_code(prompt, model_engine='text-davinci-003', max_tokens=1024, temperature=0.5, n=1, stop=None):
    """
    Generates code using the OpenAI API based on the given prompt.

    :param prompt: The input prompt for the code generation.
    :param model_engine: The OpenAI model engine to use for code generation (default: 'text-davinci-002').
    :param max_tokens: The maximum number of tokens to generate (default: 1024).
    :param temperature: The sampling temperature for the generated text (default: 0.5).
    :param n: The number of generated texts to return (default: 1).
    :param stop: The stop sequence for the generated text (default: None).
    :return: The generated code as a string.
    """
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key not set. Please set the 'OPENAI_API_KEY' environment variable.")
        
    # Initialize the OpenAI API client
    openai.api_key = api_key
    
    try:
        # Generate the code
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=stop,
            frequency_penalty=0,
            presence_penalty=0
        )
        code = response.choices[0].text
        return code
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")

def split_python_file(file_path, context, max_tokens=1024):
    function_pattern = re.compile(r"^\s*?def\s.*?\w+\(.*?\):")
    with open(file_path, "r") as file:
        lines = file.readlines()

    chunks = []
    chunk = []
    tokens = 0
    for line in lines:
        if function_pattern.match(line) and tokens > max_tokens:
            chunks.append({"context": context, "code": chunk})
            chunk = []
            tokens = 0
        chunk.append(line)
        tokens += len(line)
    chunks.append({"context": context, "code": chunk})

    return chunks

def save_chunks(chunks, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(f"{output_dir}/chunk_{i}.py", "w") as chunk_file:
            chunk_file.writelines(chunk["code"])

def generate_modified_chunks(chunks, modified_chunks_dir):
    Path(modified_chunks_dir).mkdir(parents=True, exist_ok=True)
    modified_chunks = []
    
    for i, chunk in enumerate(chunks):
        context = chunk["context"]
        code = "".join(chunk["code"])
        
        # Generate a summary of the current chunk
        summary_prompt = f"Summarize the purpose of the following python code:\n{code}"
        summary = generate_code(summary_prompt).strip()
        
        # Append the summary to the context.txt file
        with open(f"{modified_chunks_dir}/context.txt", "a") as f:
            f.write(f"Chunk {i} summary: {summary}\n\n")
            
        prompt = f"{context}\n\nRefactor and improve the code above:"
        
        modified_chunk = generate_code(prompt)
        modified_chunks.append({"input_chunk": code, "modified_chunk": modified_chunk})
        
        modified_chunk_path = f"{modified_chunks_dir}/chunk_{i}_modified.py"
        with open(modified_chunk_path, "w") as modified_chunk_file:
            modified_chunk_file.write(modified_chunk)
            
        # Remove the leading newline from the modified chunk file
        remove_leading_newline(modified_chunk_path)
        
    return modified_chunks

def combine_chunks(modified_chunks, output_path):
    with open(output_path, "w") as output_file:
        for chunk_data in modified_chunks:
            output_file.write(chunk_data["modified_chunk"])

def remove_leading_newline(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    with open(file_path, "w") as file:
        file.writelines(lines[1:] if lines and lines[0] == "\n" else lines)
        
def remove_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
        
def clean_directory(directory):
    shutil.rmtree(directory)
        
def refactor_large_code(input_file, output_file):
    input_dir = os.path.dirname(input_file)
    modified_chunks_dir = os.path.join(input_dir, "modified_chunks")
    
    context_file = os.path.join(modified_chunks_dir, "context.txt")
    remove_file(context_file)  # Remove the context.txt file before each script execution
    clean_directory(modified_chunks_dir)
    
    with open(input_file, "r", encoding="ISO-8859-1") as f:
        content = f.read()
    
    chunks = split_python_file(input_file, content)
    save_chunks(chunks, input_dir)
    modified_chunks = generate_modified_chunks(chunks, modified_chunks_dir)
    combine_chunks(modified_chunks, output_file)
    remove_leading_newline(output_file)

if __name__ == "__main__":
    input_file = input("Please enter the path to the large Python file: ")
    output_file = "refactored_python_script.py"
    
    refactor_large_code(input_file, output_file)
