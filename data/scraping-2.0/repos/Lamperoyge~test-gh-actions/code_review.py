
import subprocess
import openai
import os
from dotenv import load_dotenv

load_dotenv()

def get_changed_code():
    diff = subprocess.check_output(['git', 'diff']).decode()
    return diff


# // TODO: refactor this function to be more readable
def chunk_diff(diff, max_length):
    chunks = []
    current_chunk = ""
    for line in diff.split("\n"):
        if len(current_chunk) + len(line) > max_length:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def openai_code_review(chunks):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    feedback = ""
    for chunk in chunks:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="Perform a code review for these changes. Be specific where you can detect possible bugs, ways to improve the code, make sure it respects the best coding standards such as syntax, logic, best practices etc, DRY code, check for infinite loops and errors. For each comment, specify the file path and the line in this format: filename.py - Line 2: \n\n" + chunk,
            max_tokens=150
        )
        feedback += response.choices[0].text.strip() + "\n\n"
    return feedback

def main():
    diff = get_changed_code()
    if not diff:
        print("No uncommitted changes.")
        return
    max_length = 1000  # Adjust as needed based on OpenAI's token limits
    chunks = chunk_diff(diff, max_length)
    feedback = openai_code_review(chunks)
    print("Code Review Feedback:\n", feedback)

if __name__ == "__main__":
    main()
