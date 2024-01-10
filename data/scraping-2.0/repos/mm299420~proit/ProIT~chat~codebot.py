import openai
import os
import argparse
import glob
from clear import clear_screen

openai.api_key = "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"
loop = False;
log_file = None;

def generate_prompt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4020,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

def log_conversation(log_file, user_prompt, gpt_response):
    with open(log_file, 'a') as file:
        file.write(f"User: {user_prompt}\n")
        file.write(f"GPT-3.5 Turbo: {gpt_response}\n")

parser = argparse.ArgumentParser(description="GPT-3.5 Turbo chat application")
parser.add_argument("-c", "--chat", type=int, help="Select the chat session number")
parser.add_argument("-n", "--new", action="store_true", help="Create a new chat session")
parser.add_argument("-p", "--prompt", type=str, help="Enter your prompt")
parser.add_argument("-l", "--loop", action="store_true", help="create a loop")
args = parser.parse_args()

if args.loop:
    loop = True;
elif args.new:
    log_files = glob.glob("chat_*.log")
    chat_number = max([int(f.split("_")[1].split(".")[0]) for f in log_files]) + 1 if log_files else 1
    log_file = f"chat_{chat_number}.log"
elif args.chat:
    log_file = f"chat_{args.chat}.log"
else:
    log_files = glob.glob("chat_*.log")
    if log_files:
        chat_number = max([int(f.split("_")[1].split(".")[0]) for f in log_files])
        log_file = f"chat_{chat_number}.log"
    else:
        log_file = "chat_1.log"

print(f"Using log file: {log_file}")

in_code_mode = False
language = ""
filename = ""

conversation_start = "You are an AI language model. When the user activates 'code mode', you will help generate code snippets in the specified programming language for the given problem or task. The user can switch between regular chat mode and code mode using the !code command, set the programming language with !language <language>, and specify the file they want to create or edit with !file <filename>."
gpt_response = generate_prompt(conversation_start)
print(f"GPT-3.5 Turbo response: {gpt_response}")

if args.prompt:
    user_prompt = args.prompt
    if user_prompt.lower() == "!code":
        in_code_mode = not in_code_mode
        print("Code mode activated") if in_code_mode else print("Code mode deactivated")
    elif user_prompt.lower().startswith("!language "):
        language = user_prompt.split(" ")[1]
        print(f"Language set to: {language}")
    elif user_prompt.lower().startswith("!file "):
        filename = user_prompt.split(" ")[1]
        print(f"File set to: {filename}")
    else:
        if in_code_mode:
            prompt = f"Write a {language} code snippet to {user_prompt}"
        else:
            prompt = user_prompt

        gpt_response = generate_prompt(prompt)
        print(f"GPT-3.5 Turbo response: {gpt_response}")

        log_conversation(log_file, user_prompt, gpt_response)