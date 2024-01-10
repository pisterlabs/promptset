
#coding=utf8
#refer:https://stackoverflow.com/questions/37012469/duckduckgo-api-getting-search-results

'''
langchain offical tools with DuckDuckGo Search=>https://python.langchain.com/docs/integrations/tools/ddg

'''

from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults


search = DuckDuckGoSearchRun()
result = search.run("Obama's first name?")
print("DuckDuckGoSearchRun.result=>",result)

print("===============================")

print("===============================")
search = DuckDuckGoSearchResults()
result = search.run("Maharishi")
print("DuckDuckGoSearchResults.result=>",result)

print("===============================")
search = DuckDuckGoSearchResults()
result = search.run("What is the weather in NYC today, yesterday, and the day before?")
print("DuckDuckGoSearchResults.result=>",result)