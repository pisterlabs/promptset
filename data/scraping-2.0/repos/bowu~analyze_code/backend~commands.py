import os
import re
import shutil
from typing import Callable, Dict, Union, List
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

CHAR_LIMIT = 4000*4

embeddings = OpenAIEmbeddings()
query_engine = None

async def process_files(
    dir_path: str,
    process_file: Callable[[str], None],
    file_pattern: str = r".*\.(py|cpp|problems|comments)$"
) -> None:
    for entry in os.scandir(dir_path):
        if entry.is_file() and re.match(file_pattern, entry.name):
            await process_file(entry.path)  # Add 'await' before the callback
        elif entry.is_dir():
            await process_files(entry.path, process_file, file_pattern)

async def complete(prompt) -> str:
        if len(prompt) > CHAR_LIMIT:
            prompt = prompt[:CHAR_LIMIT]

        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": str(prompt)}
        ]
        )

        return completion.choices[0].message.content


async def generate_comments(model, file_path: str) -> str:
    with open(file_path, "r") as input_file:
        input_code = input_file.read()

        # prompt=f"""For the given C++ or Python code, generate detailed
        # documentation for each function, including its signature, a description
        # of what it does, its inputs, and its output. Do not include the function
        # body. Only output information
        # about the code.
        # Input Code:
        # def add(a, b):
        #   return a + b
        # Output:
        # Signature: add(a, b)
        # Function input: two numbers a, b
        # Function output: the sum of a and b
        # Description: Adds two numbers together.\n
        # Input Code:{input_code}""",

        prompt = f"""For the given code, generate detailed documentation for each function, including its signature, a description of what it does, and anything else that would be helpful for another engineer to understand it. Only output information about the code. Do not include the function body. A file may include many functions. Make sure that the output is function-based and easy to understand. 
        Input Code:
        {input_code}"""

        print(f"Processing {file_path} to generate comments...")
        response = await complete(prompt)

        return f"{file_path}\n\n{response}"

async def generate_problems(model, file_path: str) -> str:
    with open(file_path, "r") as input_file:
        input_code = input_file.read()

        # prompt=f"""For the given C++ or Python code, provide detailed analysis
        # of its bugs and optionally performance problems for each function. Do
        # not include the function body.
        # Input code:
        # def divide_by_n(n, nums):
        #     result = []
        #     for num in nums:
        #         result.append(num / n)
        #     return result

        # def find_duplicates(lst):
        #     duplicates = []
        #     for elem in lst:
        #         if lst.count(elem) > 1 and elem not in duplicates:
        #             duplicates.append(elem)
        #     return duplicates
        # Output:
        # Function: divide_by_n(n, nums)
        # The function has bugs: The bug in this function is that it does not
        # handle the case where n is equal to 0. If n is equal to 0, the function
        # will try to divide all of the numbers in nums by 0, which will result in
        # a ZeroDivisionError. This error can cause the function to crash, or
        # produce incorrect or unexpected results.\n  
        # Function: find_duplicates(lst)
        # The function has performance problems: The performance issue with this
        # function is that it uses the list.count() method to check how many times
        # an element appears in the list for each element in the list. This method
        # has a time complexity of O(n), where n is the length of the list.
        # Therefore, the time complexity of this function is O(n^2), which can be
        # very slow for large lists.
        # Input code:{input_code}""",

        prompt = f"""For the given code, provide detailed analysis of its bugs and optionally performance problems for each function. Do not include the function body. A file may include many functions. Make sure that the output is function-based and easy to understand.
        Input code: {input_code}"""
        print(f"Processing {file_path} to generate problems...")
        response = await complete(prompt)

        return f"{file_path}\n\n{response}"

async def process_file_with_openai(model, file_path: str, dir_path: str) -> None:
    comments = await generate_comments(model, file_path)
    problems = await generate_problems(model, file_path)

    file_name = os.path.basename(file_path)
    rel_dir_path = os.path.relpath(os.path.dirname(file_path), dir_path)
    rel_dir_path = rel_dir_path if rel_dir_path != "." else ""
    comments_path = f"{dir_path}/_comments/{rel_dir_path}/{file_name}.comments"
    problems_path = f"{dir_path}/_problems/{rel_dir_path}/{file_name}.problems"

    os.makedirs(os.path.dirname(comments_path), exist_ok=True)
    os.makedirs(os.path.dirname(problems_path), exist_ok=True)

    # Write the generated content to the respective files
    with open(comments_path, "w") as comments_file:
        comments_file.write(comments)

    with open(problems_path, "w") as problems_file:
        problems_file.write(problems)

async def generate_query_engine(path:str) -> Chroma:
    comments_path = os.path.join(path, "_comments")
    problems_path = os.path.join(path, "_problems")

    file_contents = []

    async def read_file_content(file_path: str):
        with open(file_path, "r") as file:
            content = file.read()
            file_contents.append(content)

    await process_files(comments_path, read_file_content)
    await process_files(problems_path, read_file_content)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.create_documents(
        file_contents,
        metadatas=[]
    )
    docsearch = Chroma.from_documents(documents, embeddings)
    return docsearch

async def handle_message(message, model) -> str:
    if message.startswith("/"):
        return await handle_command(message, model)
    else:
        return await handle_query(message)

async def handle_query(query: str) -> str:
    if not query_engine:
        return "You should run the inspect command first. Example: /inspect /Users/username/your_project"
    search_result = query_engine.similarity_search(query)[0].page_content
    prompt = f"""User's query is {query} and the relevant content is {search_result}. Synthesize a response to the user's query which contains only relevant information. Be polite."""
    response = await complete(prompt)    
    return response

async def handle_command(message: str, model) -> str:
    command_handlers: Dict[str, Callable[[List[str], str], Union[str, None]]] = {
        "/inspect": handle_inspect_command,
    }

    command_parts = message.split(" ")

    command = command_parts[0]

    if command in command_handlers:
        return await command_handlers[command](command_parts, model)
    else:
        return "Invalid command"


async def handle_inspect_command(command_parts: List[str], model: str) -> str:
    if len(command_parts) < 2:
        return "Error: Insufficient arguments for /inspect command. Please provide a path."

    path = command_parts[1]
    if os.path.exists(f"{path}/_comments"):
        shutil.rmtree(f"{path}/_comments")
    if os.path.exists(f"{path}/_problems"):
        shutil.rmtree(f"{path}/_problems")

    try:
        await process_files(
            path,
            lambda file_path: process_file_with_openai(model, file_path, path),
            file_pattern=r".*\.(py|cpp)$",
        )
        global query_engine
        query_engine = await generate_query_engine(path)
        return "Command completed successfully"
    except Exception as error:
        return str(error)
