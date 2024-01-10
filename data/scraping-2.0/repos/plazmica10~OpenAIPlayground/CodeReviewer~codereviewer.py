import openai
from dotenv import dotenv_values
import os
import argparse

# Prompt for the AI
prompt = """ 
    You will receive a file's contents as text. Generate a code review for the file. 
    Indicate only necessary changes to improve its style, performance readability, and maintainability.
    For each suggested change, include line numbers to which you are referring. 
    If there are any reputable libraries that could be introduced to improve the code, suggest them. 
    Be kind and constructive/
"""

#Sends a request to the OpenAI API
def make_request(filecontent,model):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Code review the following file: {filecontent}"},
    ]
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return res["choices"][0]["message"]["content"]

def code_review(file_path,model):
    # Read file contents
    with open(file_path, "r") as f:
        filecontent = f.read()
        generated_review = make_request(filecontent,model)
        print(generated_review)


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description="Code Reviewer")
    parser.add_argument("file")
    parser.add_argument("--model",default="gpt-3.5-turbo")
    args = parser.parse_args()    
    code_review(args.file,args.model)

if __name__ == "__main__":
    config = dotenv_values("../.env")
    openai.api_key = config["AIKEY"]
    main()

