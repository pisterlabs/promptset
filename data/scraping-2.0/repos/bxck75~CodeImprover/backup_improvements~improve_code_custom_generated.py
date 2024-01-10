import sys
import re
from pathlib import Path
import g4f

class CodeImprover:
    """
    A pragmatic, procedural, and organized code analyzing and improving agent.
    """

    @staticmethod
    def read_code(text: str) -> str:
        match = re.search(r"```(python|py|)\n(?P<code>[\s\S]+?)\n```", text)
        if match:
            return match.group("code")

    def improve_code(self) -> None:
        """
        Improves the code in a given file.
        """
        path = input("path: ")

        try:
            with open(path, "r") as file:
                code = file.read()

            ingesting_and_analyzing_role = """you are a pragmatic, procedural and organized codeanalyzing and codeimproving agent.
                                    its the year 3145 and your task is to ingest,analyze and improvea scripts. """

            alter_prompt = f"""
            {ingesting_and_analyzing_role}
            improve the code in this file:
            ```py
            {code}
            ```
            tasks:
                add methods, enums or classes if needed.
                add type hints if possible.
                add an llm-agent (g4fllm) to automate tasks by adding 'from langchain_g4f import G4FLLM if possible.
                refactor the code into a fully OOP callable class with __name__ == "__main__":
                look for lines starting with # todo: and try to add the functionality described there as a separate experimental try-except block under the __name__ == "__main__":
                add a usage comment block at the end of the script explaining the usage of all functionalities of the script
                add predicted use cases in a comment block
                propose extra features in a comment block
                add a pretty docstring at the beginning of the script

            rules:
                - don't make upper or lower case errors in methods or variable naming
                - don't change methods or classes that have '#protected' above the definition
                - don't remove anything, only add functionality
                - don't remove imports of the script
                - don't add any type hints to args and kwargs
                - don't remove license comments.
            """

            print(f"create code for script at path: {path}...")

            response = []
            for chunk in g4f.ChatCompletion.create(
                model=g4f.models.gpt_35_long,
                messages=[{"role": "user", "content": alter_prompt}],
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
Usage:
- Run the script.
- Enter the path of the file you want to improve when prompted.
- The script will use a language model (g4f) to generate an improved version of the code.
- The improved code will be saved in a new file with a suffix "_generated_improvement" in the file name.
"""

"""
Predicted use cases:
- Improving code by analyzing and providing automated suggestions.
- Learning and understanding the improvements made by the language model.
"""

"""
Proposed extra features:
- Add support for improving all files in a given directory.
- Add support for specifying the language model to use.
- Add support for customizing the code improvement prompt.
"""