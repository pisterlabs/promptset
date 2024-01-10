import openai
import os
import sys
import subprocess


def get_api_key():
    # Replace with your method of fetching the API key
    return os.environ.get("OPENAI_API_KEY")

def query_gpt(prompt, api_key):
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=150  # or another model
    )
    return response.choices[0].text.strip()


def execute_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"


def get_full_prompt(user_prompt):
    my_path = os.path.abspath(__file__)
    prompt_path = os.path.dirname(my_path)
    prompt_file = os.path.join(prompt_path, "prompt.txt")
    pre_prompt = open(prompt_file, "r").read()
    # pre_prompt = pre_prompt.replace("{shell}", shell)
    # pre_prompt = pre_prompt.replace("{os}", get_os_friendly_name())
    prompt = pre_prompt + user_prompt

    if prompt[-1:] != "?" and prompt[-1:] != ".":
        prompt += "?"

    return prompt


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask_gpt.py 'Your input here'")
        sys.exit(1)

    api_key = get_api_key()
    if not api_key:
        print("OpenAI API key is not set. Please set it and try again.")
        sys.exit(1)

    prompt = sys.argv[1]
    prompt = get_full_prompt(prompt)
    proposed_command = query_gpt(prompt, api_key)

    print(f"GPT proposes to run: {proposed_command}")

    confirm = input("Press Enter to execute, or type anything else to cancel: ")
    if confirm == "":
        output = execute_command(proposed_command)
        print(output)
    else:
        print("Command execution cancelled.")


if __name__ == "__main__":
    main()
