from langchain.utilities import DuckDuckGoSearchAPIWrapper

wrapper = DuckDuckGoSearchAPIWrapper()

def web_search(query, num_of_results=1):
    results = wrapper.results(query, num_of_results)
    return [r['link'] for r in results]
