import argparse
from dotenv import load_dotenv
from openai import OpenAI
import os

def get_completion(prompt, model):

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPEN_AI_KEY")
    )

    messages = [{"role": "user", "content": prompt}]

    response =  client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='AI Command Line Assistant')
    parser.add_argument('command', help='Command to process')
    args = parser.parse_args()

    home_dir = os.path.expanduser("~")
    dotenv_path = os.path.join(home_dir, ".secrets", "aicmd", ".env")
    load_dotenv(dotenv_path)

    ai_command = args.command

    prompt = f"""
    print command that '{ai_command}'.
    If user asks to search or find something, then make sure to include hidden files,  unless user specifies otherwise.
    If user doesn't specify the scope of search command, then use current directory.
    If user asks for git command make sure to do include only git command.
    When user asks to look for a pattern inside files, make sure pattern match is case-insensetive, unless user specifies otherwise.
    Output should be a single line and signle command, no quotes, no new lines.
    """

    result = get_completion(prompt, model="gpt-4")
   
    if result is not None:
        print(result)
    else:
        print("Can not generate command. Try another promt.")

if __name__ == "__main__":
    main()
