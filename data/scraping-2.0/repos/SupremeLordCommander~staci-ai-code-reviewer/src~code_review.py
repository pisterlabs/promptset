import os
import subprocess
import openai

def get_changed_files():
    try:
        # Getting a list of all changed files in the last commit
        result = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD~1', 'HEAD']).decode('utf-8')
        files = result.strip().split('\n')
        return files
    except Exception as e:
        print("Error getting changed files:", str(e))
        return []

def review_code(file):
    try:
        with open(file, 'r') as f:
            code = f.read()

        # Interact with OpenAI API to review the code
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Review the following code and suggest improvements if needed. Also, explain what the changes are attempting:\n\n{code}",
            temperature=0.5,
            max_tokens=1000,
        )

        suggestions = response.choices[0].text.strip()

        if suggestions:
            print(f"Suggestions for {file}:\n{suggestions}\n")
        else:
            print(f"No suggestions from OpenAI for {file}.\n")

    except Exception as e:
        print(f"Error reviewing file {file}:", str(e))

def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Get a list of changed files in the PR
    changed_files = get_changed_files()
    
    # Review each changed file
    for file in changed_files:
        review_code(file)

if __name__ == "__main__":
    main()
