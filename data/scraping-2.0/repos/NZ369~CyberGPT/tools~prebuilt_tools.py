from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import HumanInputRun
#from langchain.tools import ShellTool
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.agents.tools import Tool

python_repl = PythonREPL()
python_tool = Tool(
    name='python repl',
    func=python_repl.run,
    description="Useful for when you need to use python to process a CSV or JSON, or to build a machine learning model. You should input python code."
)

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
)

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet. be specific with your input. This is the least reliable tool."
)

human = HumanInputRun()
human_tool = Tool(
    name='Human Input',
    func= human.run,
    description="Useful when you're not sure what to do next and you need to clarify something with a human expert."
)

#shell = ShellTool()
#shell_tool = Tool(
#    name='Shell Commands',
#    func= shell.run,
#    description="Useful when you want to interact with the local filesystem to run shell commands."
#)
