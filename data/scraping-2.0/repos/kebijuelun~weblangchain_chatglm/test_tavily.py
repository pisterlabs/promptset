from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever

retriever = TavilySearchAPIRetriever(k=3)
print(retriever.invoke("程序员如何保护头发"))
