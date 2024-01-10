from models.models import (QueryResult, AskResult)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
async def ask_based_on_query_result(query_result: QueryResult) -> AskResult:
    logging.info(f"query_result: {query_result}")

    query = query_result.query
    premise = ""
    urls = []
    for doc_chunk in query_result.results:
        if hasattr(doc_chunk, 'text'):
            premise += doc_chunk.text
        if hasattr(doc_chunk.metadata, 'url'):
            urls.append(doc_chunk.metadata.url)

    answer = await ask_llm(query, premise)
    logging.info(f"answer: {answer}, type: {type(answer)}")
    return AskResult(query=query, answer=answer, reference_urls=urls)

chat = ChatOpenAI(temperature=0)
template = PromptTemplate.from_template("premise: {premise}. answer the following question according to the premise: {query}")
async def ask_llm(query: str, premise: str) -> str:
    prompt = template.format(query=query, premise=premise)
    return chat.predict(prompt)