from langchain.chains.summarize import load_summarize_chain


def summarise(llm, docs):
    # Runs a map/reduce process, each page is summarised (map phase)
    # then a summary of summaries is generated (reduce phase)
    # https://python.langchain.com/docs/use_cases/summarization#option-2.-map-reduce
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)
