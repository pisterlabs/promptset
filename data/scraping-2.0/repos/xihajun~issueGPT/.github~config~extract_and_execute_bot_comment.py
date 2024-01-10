import json
import openai
import os
import subprocess
import re

openai.api_key = os.getenv("OPENAI_API_KEY")


class CodeExecutor:
    def __init__(self, language: str, code: str):
        self.language = language.lower()
        self.code = code

    def execute(self):
        shebang = {
            "python": "#!/usr/bin/env python3",
            "bash": "#!/bin/sh",
            # Add more languages and shebangs here as needed
        }.get(self.language, "#!/bin/sh")

        temp_script_filename = "temp_script.sh"

        with open(temp_script_filename, "w") as f:
            f.write(f"{shebang}\n{self.code}")
        try:
            os.chmod(temp_script_filename, 0o755)
            result = subprocess.run(
                ["./" + temp_script_filename],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error executing code: {e}")
            return f"Error executing code: {e}"
        finally:
            subprocess.run(["rm", temp_script_filename])

    def generate_commit_message(self):
        conversations = generate_conversations(self.code)
        answer = generate_answer(conversations)
        return answer.strip()


def generate_conversations(code: str):
    conversations = [
        {
            "role": "system",
            "content": "You are a professional programmer and have done lots of git commits which can do code summary really precise and accurate. Provide a one-line github commit for the given code.",
        }
    ]
    conversations.append({"role": "user", "content": f"```{code}```"})
    return conversations


def generate_answer(conversations: list):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=conversations
    )
    answer = completion.choices[0].message.content
    return answer


def extract_code_from_comment(comment: str):
    code_snippets = []
    code_start = False
    code_block = ""
    language = ""

    for line in comment.split("\n"):
        if "```" in line:
            if not code_start:
                code_start = True
                language = re.sub(r"```(.*)", r"\1", line).strip()
            else:
                code_start = False
                code_snippets.append((code_block.strip(), language))
                code_block = ""
                language = ""
        elif code_start:
            code_block += line + "\n"

    return code_snippets


if __name__ == "__main__":
    with open(".github/comment-template.md", "r") as f:
        latest_bot_comment = f.read()

    if latest_bot_comment:
        code_snippets = extract_code_from_comment(latest_bot_comment)

        summaries = []
        for code, language in code_snippets:
            print(f"Executing {language} code:\n{code}")
            executor = CodeExecutor(language, code)
            code_output = executor.execute()
            summary = executor.generate_commit_message()
            summaries.append(summary)

            with open(".github/comment-template.md", "a") as f:
                f.write(f"\nOutput:\n```\n{code_output}\n```")
        summaries_str = "\n".join(summaries)
        print(f"::set-output name=summaries::{summaries_str}")
        
        
