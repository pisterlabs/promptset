from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
search.run("Obama's first name?")
