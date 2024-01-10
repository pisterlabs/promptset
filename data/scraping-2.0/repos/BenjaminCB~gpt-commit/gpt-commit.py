import subprocess
import openai
import os

def get_git_diff():
    """Get the difference between the git index and head."""
    result = subprocess.run(['git', '--no-pager', 'diff', '--cached'], capture_output=True, text=True)
    return result.stdout

def get_commit_message_from_chatgpt(diff_output):
    """Prompt ChatGPT for a commit message based on the git diff."""

    # Retrieve API key from environment variable
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set!")
        return

    openai.api_key = openai_api_key

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Based on the following git changes:\n\n" + diff_output + "\n\nProvide a suitable commit message, and only include the commit message in your response:"}]
    )

    message = completion.choices[0].message.content

    return message

def main():
    diff_output = get_git_diff()
    if not diff_output:
        print("No changes detected. Exiting.")
        return

    commit_message = get_commit_message_from_chatgpt(diff_output)
    if commit_message:
        subprocess.run(['git', 'commit', '-m', commit_message])
        print(f"Committed with message: {commit_message}")

if __name__ == "__main__":
    main()

