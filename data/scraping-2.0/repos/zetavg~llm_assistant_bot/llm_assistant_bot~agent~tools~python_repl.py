from langchain.agents import Tool
from langchain.utilities import PythonREPL


def get_python_repl_tool():
    python_repl = PythonREPL()

    async def python_repl_arun(command):
        return python_repl.run(command)

    python_repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command, and you need to import libraries before using them. You must use `print(...)` in order to see the output. Do not use this unless it's necessary.",
        func=python_repl.run,
        coroutine=python_repl_arun,
    )
    return python_repl_tool
