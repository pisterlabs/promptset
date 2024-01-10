import subprocess
import openai
import markdown, re
from PyInquirer import prompt as py_inquirer_prompt, style_from_dict, Token


def load_config(path):
    return {
        "prefix": "",
        
    }

def run_command(command):
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        raise Exception(f"Command {command} failed with exit code {process.returncode}")
    return process.stdout


def check_if_commits_are_staged():
    try:
        result = run_command("git diff --staged")
        if result == "":
            return False
    except Exception:
        return False
    return True


def generate_commit_message_from_diff(diff):
    prompt = f"""
    What follows "-------" is a git diff for a potential commit.
    Reply with a markdown unordered list of 5 possible, different Git commit messages 
    (a Git commit message should be concise but also try to describe 
    the important changes in the commit), order the list by what you think 
    would be the best commit message first, and don't include any other text 
    but the 5 messages in your response.
    ------- 
    {diff}
    -------
    """
    if len(prompt) >= 4096:
        # cut off for max prompt length
        prompt = prompt[:4000]
        
    print("Generating commit message...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    message = response["choices"][0]["message"]["content"]
    return message  # .strip().replace('"', '').replace("\n", '')


def main():
    if not check_if_commits_are_staged():
        print("No staged commits")
        exit(0)
    diff = run_command("git diff --staged")
    commit_message = generate_commit_message_from_diff(diff)

    html = markdown.markdown(commit_message)
    suggestions = re.findall(r"<li>(.*?)</li>", html)
    if len(suggestions) == 0:
        print("No suggestions found.")
        exit(0)

    # run_command(f'git commit -m "{commit_message}"')
    answers = py_inquirer_prompt(
        [
            {
                "type": "list",
                "name": "commit_message",
                "message": "Commit message suggestions:",
                "choices": [f"{i + 1}. {item}" for i, item in enumerate(suggestions)],
                "filter": lambda val: val[3:],
            }
        ]
    )
    answers = py_inquirer_prompt(
        [
            {
                "type": "input",
                "name": "final_commit_message",
                "message": "Confirm or edit the commit message:",
                "default": answers.get("commit_message"),
            },
        ]
    )
    cmt_msg = answers.get("final_commit_message")

    print(f"Committed with message: {cmt_msg}")
    run_command(f'git commit -m "{cmt_msg}"')


if __name__ == "__main__":
    main()
