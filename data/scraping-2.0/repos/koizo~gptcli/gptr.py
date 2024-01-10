#!/usr/bin/env python3
import os
import openai
import subprocess

openai_key_location = os.environ['HOME']+'/.openai_key'
messages = []

def get_key():
    with open(openai_key_location, "r") as file:
        key = file.readline().strip()
    return key

def read_conf():
    with open(openai_key_location, "r") as file:
        content = file.readline().strip()
    return content

def query_gpt(prompt):
    # add prompt to messages
    messages.append({'role':'user','content':prompt})
    # send messages to GPT-3
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # add response to messages
    messages.append({'role':'system','content':response["choices"][0]["message"]["content"]})
    # return response
    return response["choices"][0]["message"]["content"]

def help():
    print("Welcome to GPT Terminal!")
    print("Write a question or a message to the AI and it will respond.\n")
    print("Other commands:")
    print("exit: exit the program")
    print("save <filename>: save the conversation to a file")
    print("load <filename>: load a conversation from a file")
    print("clear: clear the screen")
    print("history: show the history of the conversation")
    print("run <command>: run a shell command")

def run_command(command):
    # Run the command and capture its output.
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output, error

if __name__ == "__main__":
    openai_key = read_conf()
    openai.api_key = openai_key
    
    while True:
        # Get user input
        user_input = input("$ ")
    
        if user_input.lower() == "exit":
            # Exit the program
            break
        if user_input.lower()[:4] == "save":
            with open(user_input[5:], "w") as file:
                file.write(str(messages))
                print("> Saved!")
            continue
        if user_input.lower()[:4] == "load":
            # Load the contents of the file
            with open(user_input[5:], "r") as file:
                # Read the contents of the file
                file_contents = file.read()
                # Convert the string to a list
                messages = eval(file_contents)
                print("> Loaded!")
                continue
        if user_input.lower() == "clear":
            os.system('clear')
            continue
        if user_input.lower() == "history":
            # Print the history of the conversation
            for message in messages:
                print(f"{message['role']}: {message['content']}")
            continue
        if user_input.lower()[:3] == "run":
            # Run the command and capture its output.
            output, error = run_command(user_input[4:])
            print(output.decode("utf-8"))
            continue
        if user_input.lower() == "help":
            # Print the help message
            help()
            continue

        response = query_gpt(user_input)
        print(f"> {response.strip()}")

