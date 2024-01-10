from datetime import datetime
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from functions import (
    exec_static_sql_query,
    get_arxiv_search_results, 
    get_pdf_document_chunks,
    get_postgre_db_schema,
    get_serp_links, 
    scrape_webpage_text
)
from prompts import (
    arxiv_search_queries_prompt,
    report_with_query_prompt,
    report_with_a_source_query_prompt,
    document_summarization_prompt,
    free_summarization_prompt,
    directed_summarization_prompt,
    sql_query_prompt,
    sql_query_with_result_prompt,
    web_search_engine_queries_prompt
)

generate_arxiv_search_queries = arxiv_search_queries_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser() | json.loads

generate_sql_query = sql_query_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()

generate_sql_nlp_response = RunnablePassthrough.assign(
    db_schema=lambda _: get_postgre_db_schema()
) | RunnablePassthrough.assign(
    query=generate_sql_query
) | RunnablePassthrough.assign(
    response=lambda input_obj: exec_static_sql_query(input_obj["query"])
) | sql_query_with_result_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()

generate_web_search_engine_queries = web_search_engine_queries_prompt  | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser() | json.loads

summarize_arxiv_search_result = RunnablePassthrough.assign(
    summary=RunnablePassthrough.assign(
        content=lambda input_obj: input_obj["result"]
    ) | free_summarization_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()
)

summarize_pdf_result = RunnablePassthrough.assign(
    summary=RunnablePassthrough.assign(
        content=lambda input_obj: input_obj["result"]
    ) | document_summarization_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()
)

get_and_summarize_arxiv_search_results = RunnablePassthrough.assign(
    results=lambda input: get_arxiv_search_results(input["query"])
) | (lambda list_of_results: [
    {
        "query": list_of_results["query"],
        "result": result
    } for result in list_of_results["results"]
]) | summarize_arxiv_search_result.map()

get_and_summarize_pdf_doc = RunnablePassthrough.assign(
    results=lambda input: get_pdf_document_chunks(input["source"])
) | (lambda formatted_chunks: [
    {
        "result": chunk.page_content,
        "source": formatted_chunks["source"],
    } for chunk in formatted_chunks["results"]
]) | summarize_pdf_result.map()

scrape_and_summarize_webpage = RunnablePassthrough.assign(
    summary=RunnablePassthrough.assign(
        # anonymous function that takes no arguments, and returns the output of `scrape_text`
        content=lambda input_obj: scrape_webpage_text(input_obj["url"])
    ) | directed_summarization_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()
) | (lambda summarization_res: f"URL: {summarization_res['url']}\n\nSUMMARY: {summarization_res['summary']}")

# return a list of urls based on the input query,
# then construct a dictionary list with the query and each url,
# then we apply the webpage summarization chain to each dictionary using the chain `map` method
summarize_webpages = RunnablePassthrough.assign(
    urls=lambda input: get_serp_links(input["query"])
) | (lambda list_of_urls: [
    {
        "query": list_of_urls["query"],
        "url": link
    } for link in list_of_urls["urls"]
]) | scrape_and_summarize_webpage.map()

search_arxiv = generate_arxiv_search_queries | (lambda queries: [
    {
        "query": q,
    } for q in queries
]) | get_and_summarize_arxiv_search_results.map()

# what we get here is a list of lists,
# we basically deletegate to the LLM the generation of relevant web search engine queries,
# then we visit each SERP result (up to provided limit) and summarize the page we've found
search_the_web = generate_web_search_engine_queries | (lambda queries: [
    {
        "query": q,
    } for q in queries
]) | summarize_webpages.map()

generate_arxiv_search_report = RunnablePassthrough.assign(
    date=lambda _: datetime.now().strftime('%B %d, %Y'),
    # joining the list of lists of search results into a string
    summary= search_arxiv | (lambda search_results: ["\n\n".join([f"""-----------------------------------------------------------------------------------------------------------
    ARXIV SEARCH QUERY:
                                                                                 
    {r['query']}
                                                                                 
    ARXIV SEARCH RESULT: 
                                                                                 
    {r['result']}

    ARXIV SEARCH RESULT SUMMARY:

    {r['summary']}
    -----------------------------------------------------------------------------------------------------------""" for r in sr]) for sr in search_results])
) | report_with_query_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()

generate_pdf_report = RunnablePassthrough.assign(
    date=lambda _: datetime.now().strftime('%B %d, %Y'),
    summary= get_and_summarize_pdf_doc | (lambda chunk_summaries: "\n\n".join([f"""-----------------------------------------------------------------------------------------------------------
    {s}
    -----------------------------------------------------------------------------------------------------------""" for s in chunk_summaries]))
) | report_with_a_source_query_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()

generate_web_search_report = RunnablePassthrough.assign(
    date=lambda _: datetime.now().strftime('%B %d, %Y'),
    # joining the list of lists of search results into a string
    summary= search_the_web | (lambda search_results: "\n".join([f"---\n{' '.join([text for text in sr])}\n---" for sr in search_results]))
) | report_with_query_prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()