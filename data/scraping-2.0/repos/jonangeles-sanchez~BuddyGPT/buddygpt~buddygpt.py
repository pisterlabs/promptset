"""Main module."""
import openai
import click

def messenger(msg,files):
    """Console script for buddygpt."""
    connect_to_chatGPT(msg, files)
    
    return 0

def connect_to_chatGPT(msg, files):
    openai.api_key = "sk-"
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[structure_message(msg, files)])
    print_message(completion)

def extract_file_content(files):
    file_contents = []
    for file in files:
        with open(file, "r") as f:
            file_contents.append({"name" : file, "content" : f.read()})
    return file_contents

def structure_message(msg, files):
    file_contents = extract_file_content(files)
    return {"role" : "assistant", "content" : msg + "\n" + "\n".join([file["name"] + "\n" + file["content"] for file in file_contents])}

def print_message(completion):
    print(completion.choices[0].message.content)
