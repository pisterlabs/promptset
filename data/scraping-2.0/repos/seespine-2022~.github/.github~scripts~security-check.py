import os
import openai
import json
import subprocess
import ast
import astunparse


class FunctionAndClassVisitor(ast.NodeVisitor):
    def __init__(self):
        self.items = []

    def visit_ClassDef(self, node):
        self.items.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.items.append(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.items.append(node)
        self.generic_visit(node)


def get_changed_files():
    default_branch = os.environ["GITHUB_DEFAULT_BRANCH"]

    # Fetch the default branch
    subprocess.check_output(["git", "fetch", "origin", default_branch])

    command = ["git", "diff", "--name-only", f"origin/{default_branch}"]
    output = subprocess.check_output(command)
    print(output)
    changed_files = output.decode("utf-8").split("\n")
    return [file for file in changed_files if file.strip() != ""]


def extract_functions_and_classes(code):
    tree = ast.parse(code)
    visitor = FunctionAndClassVisitor()
    visitor.visit(tree)
    return [astunparse.unparse(node) for node in visitor.items]


def analyze_code_security(code):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You are our security and reliability engineer. Please critique the code that is sent to you in a professional manner. Always state the issue in a bullet point (use as heading: ### Security concerns) and then suggest an improvement (use as heading: ### Suggestion). Finally print the improved code block (use as heading: ### Improved code). Start the codeblack with ``` and close it with ```. Is there a major security issue with the following code:\n\n"
                + code
                + "\n Begin your response with stating whether or not there are any major security concerns, by saying 'No security concerns' or 'Security concerns'.",
            },
        ],
        temperature=0.5,
    )
    # print(response)

    message = response.choices[0].message.content.strip()
    print(message)

    if "no security concerns" in message.lower():
        return True, message
    else:
        return False, message


def main():
    changed_files = get_changed_files()
    has_no_concerns = True
    message = ""
    i = 0

    for file in changed_files:
        print(changed_files)
        with open(file, "r") as f:
            code = f.read()

        items = extract_functions_and_classes(code)
        print(items)

        for item in items[:]:
            is_secure, analysis = analyze_code_security(item)

            if not is_secure:
                i += 1
                has_no_concerns = False
                message += f"### {i}. File: {file}\n### Affected code:\n```{item}```\n{analysis}\n---------------\n\n"

    if has_no_concerns:
        print("All files passed the security check.")
    else:
        print("Security issues found:")
        print(message)


if __name__ == "__main__":
    main()
