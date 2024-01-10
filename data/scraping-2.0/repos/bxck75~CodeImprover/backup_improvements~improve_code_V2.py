import sys
import re
from pathlib import Path
from langchain_g4f import G4FLLM
from typing import Optional
import g4f
from g4f import ChatCompletion, models

class CodeImprover:
    """
    A pragmatic, procedural, and organized code analyzing and improving agent.
    """
    script_path: Optional[str] = None

    @staticmethod
    def read_code(text: str) -> str:
        match = re.search(r"```(python|py|)\n(?P<code>[\s\S]+?)\n```", text)
        if match:
            return match.group("code")
        
    #protected
    def improve_code(self) -> None:
        """ Improves the code in a given file. """
        if self.script_path and os.path.exsts(self.script_path):
            path = self.script_path
        else:
            path = input("path: ")

        try:
            with open(path, "r") as file:
                code = file.read()

            ingesting_and_analyzing_role = f"""
            You are a pragmatic, procedural and organized codeanalyzing and improving agent.
            Its the year 3145 and your task is to ingest, analyze and improve scripts.

            Rules:
                - don't make a new comment block at the end of the script if one exists, just add subtract info as needed
                - don't change methods or classes that have '#protected' above the definition
                - don't remove anything, only add functionality
                - don't remove imports of the script
                - all new and experimental code should be inside a try-except block
                - don't add any type hints to kwargs
                - don't remove license comments."""

            prompt = f"""
            {ingesting_and_analyzing_role}
            
            Tasks:
                add methods, enums or classes if needed.
                add type hints if possible.
                add an llm-agent (g4fllm) to automate tasks by adding 'from langchain_g4f import G4FLLM' if possible.
                refactor the code into a fully OOP callable class with __name__ == "__main__":
                look for lines starting with '# TODO:' and try to add the functionality described there.
                add a usage comment block at the end of the script explaining the usage of all functionalities of the script.
                add predicted use cases in a comment block'
                propose extra features in a comment block
                add a pretty docstring at the beginning of the script

            Now improve the code in this file:
            ```py
            {code}
            ```
            """

            print(f"create code for script at path: {path}...")

            response = []
            for chunk in ChatCompletion.create(
                model= models.gpt_35_turbo,
                messages=[{"role": "user", "content": prompt}],
                timeout=3000,
                stream=False
            ):
                response.append(chunk)
                print(chunk, end="", flush=True)

            response = "".join(response)

            code = self.read_code(response)
            if code:
                with open(path.replace(".", "_generated_improvement."), "w") as file:
                    file.write(code)
        except ValueError:
            print("Invalid file path.")

if __name__ == "__main__":
    CodeImprover().improve_code()

"""
A pragmatic, procedural, and organized code analyzing and improving agent.

This script allows you to improve the code in a given file using a language model. The improved code will be generated based on the provided code and saved in a new file with a suffix "_generated_improvement" in the file name.

Usage:
- Run the script.
- Enter the path of the file you want to improve when prompted.
- The script will use the language model to generate an improved version of the code.
- The improved code will be saved in a new file with a suffix "_generated_improvement" in the file name.

Predicted use cases:
- Improving code by analyzing and providing automated suggestions.
- Learning and understanding the improvements made by the language model.

Proposed extra features:
# TODO: Add support for improving all files in a given directory.
- Add support for specifying the language model to use.
# TODO: Add support for customizing the code improvement prompt.
"""