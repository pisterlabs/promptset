from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

search.run("Obama's first name?")
