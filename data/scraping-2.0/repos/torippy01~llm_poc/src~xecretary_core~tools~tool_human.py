from langchain.agents import load_tools
from langchain.tools import BaseTool


def create_HumanTool() -> BaseTool:
    def get_input() -> str:
        print("Insert your text.")
        print("Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")

        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "q":
                break
            contents.append(line)
        return "\n".join(contents)

    return load_tools(["human"], input_func=get_input)[0]
