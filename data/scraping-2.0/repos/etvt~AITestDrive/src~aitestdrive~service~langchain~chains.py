from operator import itemgetter
from typing import Any

from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from aitestdrive.service.langchain.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT


def create_conversational_qa_chain(llm: Runnable[Any, Any], retriever: VectorStoreRetriever):
    chat_history_chain = (
            itemgetter("chat_history")
            | RunnableLambda(lambda h: get_buffer_string(h) or "no history"))

    condense_question_chain = (
            RunnablePassthrough.assign(
                chat_history=chat_history_chain
            )
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser())

    retrieve_documents_chain = (
            itemgetter("standalone_question")
            | retriever
            | _combine_documents)

    conversational_qa_chain = (
            RunnableParallel(
                standalone_question=condense_question_chain
            )
            | {
                "context": retrieve_documents_chain,
                "question": itemgetter("standalone_question"),
            }
            | QA_PROMPT
            | llm)

    return conversational_qa_chain


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
